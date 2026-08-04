use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use tracing_subscriber::{fmt, EnvFilter, prelude::*};

mod maf;
mod est_sfs;

use clap::Parser;
use flate2::Compression;
use flate2::write::GzEncoder;
use maf::{AlignmentBlock, MafError, MafReader};
use noodles_vcf as vcf;
use noodles_vcf::variant::io::Write as VcfWrite;
use noodles_vcf::variant::record::AlternateBases;
use est_sfs::{
    Base, CollectedData, ConfigKey, ConfigRow, PolarizationMethod, PosKey, base_to_counts,
    base_to_index, base_to_states, fit_substitution_model, fit_usfs_minor_count,
    parsimony_ancestral, posterior_ancestral_per_site,
};
use rand::Rng;
use smallvec::SmallVec;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fs::{File, OpenOptions, create_dir_all, remove_file};
use std::io::{self, BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;
use tracing::{info, warn};
use vcf::variant::record::info::field::Value;
use vcf::variant::record::info::field::value::Array;

/*#[derive(Parser)]
#[command(name = "polarize")]
#[command(about = "Polarize VCFs or sort MAFs by reference contig/position")]
struct Cli {
    /// Log verbosity (info, debug, warn, error)
    #[arg(long, default_value = "info")]
    verbosity: String,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Parser)]
enum Commands {
    /// Polarize a VCF using a contig/pos-sorted MAF via streaming merge-join
    Polarize(PolarizeArgs),

    /// Sort a MAF by reference contig and position
    SortMaf(SortMafArgs),
}*/

#[derive(Parser)]
struct PolarizeArgs {
    /// Input MAF file (sorted by contig/pos). Can be gzipped.
    #[arg(short, long, value_name = "alignment.maf[.gz]")]
    maf: PathBuf,

    /// Input VCF file (sorted by contig/pos). Can be gzipped.
    #[arg(short, long, value_name = "variants.vcf[.gz]")]
    vcf: PathBuf,

    /// Comma-separated list of outgroups, e.g: panTro4,ponAbe2,gorGor6
    #[arg(short = 's', long)]
    outgroup: String,

    /// Method to polarize VCF file
    #[arg(short = 'm', long, value_enum)]
    method: Option<PolarizationMethod>,

    /// Number of random starts
    #[arg(short = 'n', long, default_value_t = 10, value_name = "N")]
    nrandom: usize,

    /// Output VCF path
    #[arg(short, long, value_name = "output.vcf.gz")]
    output: Option<PathBuf>,
}

#[derive(Parser)]
struct SortMafArgs {
    /// Input MAF file to sort (can be gzipped)
    #[arg(short, long, value_name = "input.maf[.gz]")]
    input: PathBuf,

    /// Output sorted MAF file (can be gzipped)
    #[arg(short, long, value_name = "sorted.maf[.gz]")]
    output: PathBuf,

    /// Optional contig order file (one contig per line, or .fai)
    #[arg(long, value_name = "contigs.txt")]
    contig_order: Option<PathBuf>,

    /// Maximum bytes held in memory before spilling to disk
    #[arg(long, default_value_t = 268_435_456)]
    chunk_bytes: usize,

    /// Temporary directory for sort chunks
    #[arg(long)]
    tmp_dir: Option<PathBuf>,
}


#[derive(Parser)]
struct SplitMafArgs {
    /// Input MAF file (can be gzipped)
    #[arg(short, long, value_name = "input.maf[.gz]")]
    input: PathBuf,

    /// Output MAF file (can be gzipped)
    #[arg(short, long, value_name = "output.maf[.gz]")]
    output: PathBuf,

    /// Target contig name for the reference species (e.g., 'chr1')
    #[arg(short, long)]
    contig: String,
}

struct SeqCache {
    species_idx: usize,
    chrom: String,
    strand: bool,
    bases: Vec<u8>,
    pos_by_col: Vec<i32>,
}

struct MafBlockLite {
    ref_contig: String,
    ref_start: u32,
    ref_size: u32,
    ref_pos_to_col: Vec<usize>,
    seqs: Vec<SeqCache>,
    outgroup_seq_idx: Vec<Option<usize>>,
}

impl MafBlockLite {
    fn ref_end(&self) -> u32 {
        self.ref_start + self.ref_size
    }
}

struct MafBlockIter {
    reader: MafReader,
    outgroup_index: HashMap<String, usize>,
    species_index: HashMap<String, usize>,
    n_outgroups: usize,
}

impl MafBlockIter {
    fn new(
        path: &Path,
        outgroup_index: HashMap<String, usize>,
        species_index: HashMap<String, usize>,
    ) -> Result<Self, MafError> {
        let mut reader = MafReader::from_file(path)?;
        reader.read_header()?;
        let n_outgroups = outgroup_index.len();
        Ok(Self {
            reader,
            outgroup_index,
            species_index,
            n_outgroups,
        })
    }

    fn next_block_lite(&mut self) -> Result<Option<MafBlockLite>, MafError> {
        loop {
            match self.reader.next_block()? {
                Some(block) => {
                    if let Some(lite) = block_to_lite(
                        block,
                        &self.outgroup_index,
                        &self.species_index,
                        self.n_outgroups,
                    ) {
                        return Ok(Some(lite));
                    }
                    continue;
                }
                None => return Ok(None),
            }
        }
    }
}

struct MafCursor {
    iter: MafBlockIter,
    current: Option<MafBlockLite>,
}

impl MafCursor {
    fn new(
        path: &Path,
        outgroup_index: HashMap<String, usize>,
        species_index: HashMap<String, usize>,
    ) -> Result<Self, MafError> {
        let mut iter = MafBlockIter::new(path, outgroup_index, species_index)?;
        let current = iter.next_block_lite()?;
        Ok(Self { iter, current })
    }

    fn advance(&mut self) -> Result<(), MafError> {
        self.current = self.iter.next_block_lite()?;
        Ok(())
    }

    fn block_for(
        &mut self,
        contig: &str,
        pos0: u32,
        contig_rank: &HashMap<String, usize>,
    ) -> Result<Option<&MafBlockLite>, MafError> {
        enum Decision {
            Advance,
            ReturnNone,
            ReturnSome,
        }

        loop {
            let decision = match self.current.as_ref() {
                None => return Ok(None),
                Some(block) => match contig_cmp(&block.ref_contig, contig, contig_rank) {
                    Ordering::Less => Decision::Advance,
                    Ordering::Greater => Decision::ReturnNone,
                    Ordering::Equal => {
                        if pos0 < block.ref_start {
                            Decision::ReturnNone
                        } else if pos0 >= block.ref_end() {
                            Decision::Advance
                        } else {
                            Decision::ReturnSome
                        }
                    }
                },
            };

            match decision {
                Decision::Advance => {
                    self.advance()?;
                }
                Decision::ReturnNone => return Ok(None),
                Decision::ReturnSome => return Ok(self.current.as_ref()),
            }
        }
    }
}

fn block_to_lite(
    block: AlignmentBlock,
    outgroup_index: &HashMap<String, usize>,
    species_index: &HashMap<String, usize>,
    n_outgroups: usize,
) -> Option<MafBlockLite> {
    let ref_seq = block.sequences.first()?;
    let (_ref_species, ref_contig) = split_src(&ref_seq.src);
    if ref_contig.is_empty() {
        warn!(
            "Skipping MAF block with empty reference contig: {}",
            ref_seq.src
        );
        return None;
    }

    let ref_start: u32 = ref_seq.start.try_into().ok()?;
    let ref_size: u32 = ref_seq.size.try_into().ok()?;
    if ref_size == 0 {
        return None;
    }

    let ref_pos_to_col = build_ref_pos_to_col(&ref_seq.text, ref_size);
    if ref_pos_to_col.len() != ref_size as usize {
        warn!(
            "Reference ungapped length mismatch for {}:{} (expected {}, got {})",
            ref_contig,
            ref_start,
            ref_size,
            ref_pos_to_col.len()
        );
    }

    let mut outgroup_seq_idx: Vec<Option<usize>> = vec![None; n_outgroups];
    let mut seqs: Vec<SeqCache> = Vec::with_capacity(block.sequences.len());

    for seq in &block.sequences {
        let (species, chrom) = split_src(&seq.src);
        let species_idx = match species_index.get(species) {
            Some(&idx) => idx,
            None => {
                warn!("Skipping unknown species {} in positions output", species);
                continue;
            }
        };

        let start: u32 = match seq.start.try_into() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let size: u32 = match seq.size.try_into() {
            Ok(v) => v,
            Err(_) => continue,
        };

        let pos_by_col = build_pos_by_col(&seq.text, start, size);
        let bases = seq.text.as_bytes().to_vec();
        let strand = matches!(seq.strand, maf::Strand::Reverse);

        let seq_index = seqs.len();
        seqs.push(SeqCache {
            species_idx,
            chrom: chrom.to_string(),
            strand,
            bases,
            pos_by_col,
        });

        if let Some(&og_idx) = outgroup_index.get(species) {
            outgroup_seq_idx[og_idx] = Some(seq_index);
        }
    }

    Some(MafBlockLite {
        ref_contig: ref_contig.to_string(),
        ref_start,
        ref_size,
        ref_pos_to_col,
        seqs,
        outgroup_seq_idx,
    })
}

fn build_ref_pos_to_col(ref_seq: &str, ref_size: u32) -> Vec<usize> {
    let mut map: Vec<usize> = Vec::with_capacity(ref_size as usize);
    for (i, b) in ref_seq.as_bytes().iter().enumerate() {
        if *b != b'-' && *b != b'.' {
            map.push(i);
        }
    }
    map
}

fn build_pos_by_col(seq: &str, start: u32, size: u32) -> Vec<i32> {
    let mut out: Vec<i32> = Vec::with_capacity(seq.len());
    let mut pos: u32 = start;
    let mut n_bases: u32 = 0;

    for b in seq.as_bytes() {
        if *b == b'-' || *b == b'.' {
            out.push(-1);
        } else {
            out.push(pos as i32);
            pos += 1;
            n_bases += 1;
        }
    }

    assert!(
        n_bases == size,
        "Expected {} ungapped bases, saw {}",
        size,
        n_bases
    );
    assert!(pos == start + size);

    out
}

fn split_src(src: &str) -> (&str, &str) {
    let mut parts = src.splitn(2, '.');
    let species = parts.next().unwrap_or("");
    let chrom = parts.next().unwrap_or("");
    (species, chrom)
}

fn contig_cmp(a: &str, b: &str, ranks: &HashMap<String, usize>) -> Ordering {
    match (ranks.get(a), ranks.get(b)) {
        (Some(ra), Some(rb)) => ra.cmp(rb),
        (None, Some(_)) => Ordering::Less,
        (Some(_), None) => Ordering::Greater,
        (None, None) => a.cmp(b),
    }
}

fn ensure_contig_rank(ranks: &mut HashMap<String, usize>, contig: &str) {
    if !ranks.contains_key(contig) {
        let next = ranks.len();
        ranks.insert(contig.to_string(), next);
    }
}

fn base_from_byte(b: u8) -> Base {
    match b {
        b'A' | b'a' => Base::A,
        b'C' | b'c' => Base::C,
        b'G' | b'g' => Base::G,
        b'T' | b't' => Base::T,
        b'-' | b'.' | b'N' | b'n' => Base::N,
        _ => Base::N,
    }
}

fn base_to_string(b: Base) -> &'static str {
    match b {
        Base::A => "A",
        Base::C => "C",
        Base::G => "G",
        Base::T => "T",
        Base::N => "N",
    }
}

fn outgroup_bases_for_variant(block: &MafBlockLite, pos0: u32) -> Option<(usize, Vec<Base>)> {
    let offset = pos0.checked_sub(block.ref_start)?;
    if offset >= block.ref_size {
        return None;
    }
    let col_idx = *block.ref_pos_to_col.get(offset as usize)?;
    let mut out_bases: Vec<Base> = Vec::with_capacity(block.outgroup_seq_idx.len());

    for seq_idx_opt in &block.outgroup_seq_idx {
        if let Some(seq_idx) = seq_idx_opt {
            let seq = block.seqs.get(*seq_idx)?;
            let b = seq.bases.get(col_idx).copied().unwrap_or(b'N');
            out_bases.push(base_from_byte(b));
        } else {
            out_bases.push(Base::N);
        }
    }

    Some((col_idx, out_bases))
}

fn parse_outgroups_ordered(s: &str) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut names = Vec::new();

    for raw in s.split(',').map(|x| x.trim()).filter(|x| !x.is_empty()) {
        if seen.insert(raw.to_string()) {
            names.push(raw.to_string());
        }
    }
    names
}

fn default_output_vcf(vcf: &Path) -> PathBuf {
    let mut out = vcf.to_path_buf();
    let fname = out
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("output.vcf.gz");

    let base = fname
        .strip_suffix(".vcf.gz")
        .or_else(|| fname.strip_suffix(".vcf"))
        .unwrap_or(fname);

    out.set_file_name(format!("{base}.polarized.vcf.gz"));
    out
}

fn derived_path(vcf: &Path, suffix: &str) -> PathBuf {
    let mut p = vcf.to_path_buf();
    let fname = p
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("output.vcf.gz");
    let base = fname
        .strip_suffix(".vcf.gz")
        .or_else(|| fname.strip_suffix(".vcf"))
        .unwrap_or(fname);
    p.set_file_name(format!("{base}.{suffix}"));
    p
}

/// What a single pass over the VCF tells us.
///
/// `run_polarize` already paid for this pass to get `count`, so the ordering and contig
/// checks ride along for free.
struct VcfScan {
    /// Total records, counted exactly as the old `count_variants` did.
    count: usize,
    /// Reference contigs, in the order first seen.
    contigs: Vec<String>,
    /// First place the stream went backwards, as (contig, previous_pos, this_pos).
    disorder: Option<(String, u32, u32)>,
}

fn scan_vcf(vcf_path: &Path) -> Result<VcfScan, Box<dyn std::error::Error>> {
    let mut reader = vcf::io::reader::Builder::default().build_from_path(vcf_path)?;
    reader.read_header()?;

    let mut count: usize = 0;
    let mut contigs: Vec<String> = Vec::new();
    let mut disorder: Option<(String, u32, u32)> = None;
    let mut last_contig: Option<String> = None;
    let mut last_pos: u32 = 0;

    for row in reader.records() {
        let record = row?;
        count += 1;

        let contig = record.reference_sequence_name().to_string();

        // Mirror the polarize loops: a record without a parseable start is skipped, not fatal.
        let pos = match record.variant_start() {
            Some(Ok(p)) => p.get() as u32,
            _ => continue,
        };

        match &last_contig {
            Some(prev) if prev == &contig => {
                if pos < last_pos && disorder.is_none() {
                    disorder = Some((contig.clone(), last_pos, pos));
                }
            }
            _ => {
                if !contigs.contains(&contig) {
                    contigs.push(contig.clone());
                }
                last_contig = Some(contig);
            }
        }

        last_pos = pos;
    }

    Ok(VcfScan {
        count,
        contigs,
        disorder,
    })
}

/// Report what the merge-join actually achieved.
///
/// Variants with no usable MAF alignment are dropped rather than written, so the gap between
/// `total` and `written` is MAF coverage -- the visible symptom of a MAF that does not span
/// the VCF. Without this the run just looks successful.
///
/// `flipped` counts records whose REF/ALT were swapped. It is NOT a success rate: a site
/// whose REF is already the ancestral allele is polarized correctly and is never swapped,
/// and `parsimony_ancestral` returns `None` for both that case and the undetermined one
/// (see its "ref is ancestral or unknown" contract). So zero flips is unremarkable and is
/// deliberately not warned about.
fn report_polarization(flipped: usize, written: usize, total: usize) {
    let coverage = if total == 0 {
        0.0
    } else {
        100.0 * written as f64 / total as f64
    };

    info!(
        "Processed {} variants: {} had a usable MAF alignment ({:.1}%), of which {} had \
         REF/ALT swapped to the inferred ancestral allele; {} skipped (no usable MAF alignment)",
        total,
        written,
        coverage,
        flipped,
        total.saturating_sub(written)
    );

    if written == 0 {
        warn!(
            "No variants had a usable MAF alignment: the MAF covers none of the VCF's \
             positions. Check that both files describe the same contig and assembly."
        );
    }
}

fn compute_report_every(total_variants: usize) -> usize {
    if total_variants > 1_000_000 {
        1_000_000
    } else if total_variants >= 100_000 {
        100_000
    } else {
        10_000
    }
}

fn positions_output_path(
    vcf: &Path,
    species_names: &[String],
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let first = species_names.first().ok_or("No species found in MAF")?;
    let parent = vcf.parent().unwrap_or(Path::new("."));
    Ok(parent.join(format!("{}_positions.tsv.gz", first)))
}

fn open_writer(path: &Path) -> io::Result<Box<dyn Write>> {
    let file = File::create(path)?;
    let writer: Box<dyn Write> = if path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("gz"))
        .unwrap_or(false)
    {
        Box::new(GzEncoder::new(file, Compression::default()))
    } else {
        Box::new(file)
    };
    Ok(Box::new(BufWriter::new(writer)))
}

fn write_positions_header(writer: &mut dyn Write, species_names: &[String]) -> io::Result<()> {
    write!(writer, "ref_contig\tref_pos")?;
    for name in species_names {
        write!(writer, "\t{}", name)?;
    }
    writeln!(writer)?;
    Ok(())
}

fn write_positions_row(
    writer: &mut dyn Write,
    contig: &str,
    pos1: u32,
    block: &MafBlockLite,
    col_idx: usize,
    species_count: usize,
) -> io::Result<()> {
    let mut row_bases: Vec<u8> = vec![b'N'; species_count];
    let mut row_chrom: Vec<Option<&str>> = vec![None; species_count];
    let mut row_pos: Vec<i32> = vec![-1; species_count];
    let mut row_strand: Vec<bool> = vec![false; species_count];

    for seq in &block.seqs {
        if seq.species_idx >= species_count {
            continue;
        }
        row_chrom[seq.species_idx] = Some(seq.chrom.as_str());
        row_strand[seq.species_idx] = seq.strand;
        if let Some(base) = seq.bases.get(col_idx) {
            row_bases[seq.species_idx] = *base;
        }
        if let Some(pos) = seq.pos_by_col.get(col_idx) {
            row_pos[seq.species_idx] = *pos;
        }
    }

    write!(writer, "{}\t{}", contig, pos1)?;
    for i in 0..species_count {
        let chrom = row_chrom[i].unwrap_or("");
        let base = row_bases[i] as char;
        write!(
            writer,
            "\t{},{},{},{}",
            chrom, row_pos[i], base, row_strand[i]
        )?;
    }
    writeln!(writer)?;
    Ok(())
}

/// What a single pass over the MAF tells us.
struct MafScan {
    /// Species, in the order first seen. The reference species is first.
    species: Vec<String>,
    /// Reference contigs, in the order first seen.
    contigs: Vec<String>,
}

/// Walk every block once, collecting the species order and the set of reference contigs.
///
/// The species order alone would be known after roughly one block, but the contig set needs
/// the full file, and `run_polarize` was already paying for that full pass.
fn scan_maf(maf_path: &Path) -> Result<MafScan, Box<dyn std::error::Error>> {
    let mut reader = MafReader::from_file(maf_path)?;
    reader.read_header()?;

    let mut seen: HashSet<String> = HashSet::new();
    let mut names: Vec<String> = Vec::new();
    let mut contigs: Vec<String> = Vec::new();

    while let Some(block) = reader.next_block()? {
        // The first `s` line is the reference row, which is what block_for compares against.
        if let Some(ref_seq) = block.sequences.first() {
            let (_, contig) = split_src(&ref_seq.src);
            if !contig.is_empty() && !contigs.contains(&contig.to_string()) {
                contigs.push(contig.to_string());
            }
        }

        for seq in block.sequences {
            let (species, _) = split_src(&seq.src);
            if species.is_empty() {
                continue;
            }
            if seen.insert(species.to_string()) {
                names.push(species.to_string());
            }
        }
    }

    Ok(MafScan {
        species: names,
        contigs,
    })
}

fn run_polarize(args: PolarizeArgs) -> Result<(), Box<dyn std::error::Error>> {

    let outgroup_names = parse_outgroups_ordered(&args.outgroup);
    if outgroup_names.is_empty() {
        return Err("No outgroups provided".into());
    }

    let maf_scan = scan_maf(&args.maf)?;
    let species_names = maf_scan.species;
    if species_names.is_empty() {
        return Err("No species found in MAF".into());
    }

    if maf_scan.contigs.len() > 1 {
        return Err(format!(
            "MAF {} contains {} reference contigs ({}{}). Polarization is a streaming \
             merge-join and requires a single-contig MAF. Split it first:\n  \
             flexsweep split-maf --maf {} --contig <name>",
            args.maf.display(),
            maf_scan.contigs.len(),
            maf_scan.contigs.iter().take(3).cloned().collect::<Vec<_>>().join(", "),
            if maf_scan.contigs.len() > 3 { ", ..." } else { "" },
            args.maf.display()
        )
        .into());
    }

    let mut species_index: HashMap<String, usize> = HashMap::new();
    for (i, name) in species_names.iter().enumerate() {
        species_index.insert(name.clone(), i);
    }

    let mut outgroup_index: HashMap<String, usize> = HashMap::new();
    for (i, name) in outgroup_names.iter().enumerate() {
        outgroup_index.insert(name.clone(), i);
    }

    let vcf_out = args.output.unwrap_or_else(|| default_output_vcf(&args.vcf));

    match args.method {
        Some(m) => {
            info!(
                "Running {:?} polarization. Polarized alleles will be written at {:?}",
                m, &vcf_out
            );
        }
        None => {
            warn!(
                "No polarization method selected using {:?} two-parameter fit by default",
                PolarizationMethod::Kimura
            );
        }
    }

    let total_time = Instant::now();
    let vcf_scan = scan_vcf(&args.vcf)?;
    let total_variants = vcf_scan.count;
    let count_time = total_time.elapsed();
    info!("Time to count variants: {:?}", count_time);

    // The merge-join cursor is forward-only and never rewinds: on unsorted or multi-contig
    // input it silently overshoots and drops variants. Refuse rather than return a quietly
    // incomplete VCF.
    if vcf_scan.contigs.len() > 1 {
        return Err(format!(
            "VCF {} contains {} contigs ({}{}). Polarization requires a single-contig VCF. \
             Extract one first:\n  \
             bcftools view -r <name> {} -Oz -o <name>.vcf.gz && tabix -p vcf <name>.vcf.gz",
            args.vcf.display(),
            vcf_scan.contigs.len(),
            vcf_scan.contigs.iter().take(3).cloned().collect::<Vec<_>>().join(", "),
            if vcf_scan.contigs.len() > 3 { ", ..." } else { "" },
            args.vcf.display()
        )
        .into());
    }

    if let Some((contig, prev, cur)) = &vcf_scan.disorder {
        return Err(format!(
            "VCF {} is not position-sorted: {}:{} follows {}:{}. Sort it first:\n  \
             bcftools sort {} -Oz -o sorted.vcf.gz && tabix -p vcf sorted.vcf.gz",
            args.vcf.display(),
            contig,
            cur,
            contig,
            prev,
            args.vcf.display()
        )
        .into());
    }

    // A name mismatch (MAF 'CM000663.2' vs VCF 'chr1') makes every contig_cmp miss, which
    // would otherwise finish cleanly with zero polarizations and exit code 0.
    match (maf_scan.contigs.first(), vcf_scan.contigs.first()) {
        (Some(maf_contig), Some(vcf_contig)) if maf_contig != vcf_contig => {
            return Err(format!(
                "Contig mismatch: MAF {} has '{}' but VCF {} has '{}'. The reference contig \
                 must be named identically in both files.",
                args.maf.display(),
                maf_contig,
                args.vcf.display(),
                vcf_contig
            )
            .into());
        }
        (_, None) => {
            return Err(format!("VCF {} contains no variants", args.vcf.display()).into());
        }
        _ => {}
    }

    let report_every = compute_report_every(total_variants);

    let method = args.method.unwrap_or(PolarizationMethod::Kimura);
    let positions_out = positions_output_path(&args.vcf, &species_names)?;

    match method {
        PolarizationMethod::Parsimony => run_parsimony(
            &args.maf,
            &args.vcf,
            &vcf_out,
            &positions_out,
            &species_names,
            &species_index,
            &outgroup_index,
            total_variants,
            report_every,
        )?,
        _ => run_model_based(
            &args.maf,
            &args.vcf,
            &vcf_out,
            &positions_out,
            &species_names,
            &species_index,
            &outgroup_names,
            &outgroup_index,
            method,
            args.nrandom,
            total_variants,
            report_every,
        )?,
    }

    Ok(())
}

fn run_parsimony(
    maf_path: &Path,
    vcf_path: &Path,
    vcf_out: &Path,
    positions_out: &Path,
    species_names: &[String],
    species_index: &HashMap<String, usize>,
    outgroup_index: &HashMap<String, usize>,
    total_variants: usize,
    report_every: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut positions_writer = open_writer(positions_out)?;
    write_positions_header(&mut *positions_writer, species_names)?;

    let mut contig_rank: HashMap<String, usize> = HashMap::new();
    let mut maf_cursor = MafCursor::new(maf_path, outgroup_index.clone(), species_index.clone())?;

    let mut reader = vcf::io::reader::Builder::default().build_from_path(vcf_path)?;
    let header = reader.read_header()?;

    let mut writer = vcf::io::writer::Builder::default().build_from_path(vcf_out)?;
    writer.write_header(&header)?;

    let mut processed: usize = 0;
    let mut flipped_count: usize = 0;
    let mut written_count: usize = 0;
    let total_time = Instant::now();
    let species_count = species_names.len();

    info!("Polarizing variants 0/{}", total_variants);
    for row in reader.records() {
        let record = row?;
        processed += 1;
        if processed % report_every == 0 || processed == total_variants {
            info!("Polarizing variants {}/{}", processed, total_variants);
        }

        let contig = record.reference_sequence_name().to_string();
        ensure_contig_rank(&mut contig_rank, &contig);

        let pos1: u32 = match record.variant_start() {
            Some(Ok(p)) => p.get() as u32,
            _ => continue,
        };
        let pos0 = pos1.saturating_sub(1);

        let block = match maf_cursor.block_for(&contig, pos0, &contig_rank)? {
            Some(b) => b,
            None => continue,
        };

        if block.seqs.len() < outgroup_index.len() {
            continue;
        }

        let (col_idx, out_bases) = match outgroup_bases_for_variant(block, pos0) {
            Some(v) => v,
            None => continue,
        };

        let mut outgroup_alleles: HashSet<String> = HashSet::new();
        for b in &out_bases {
            outgroup_alleles.insert(base_to_string(*b).to_string());
        }
        if outgroup_alleles.len() > 1 {
            continue;
        }

        let ref_allel = record.reference_bases().to_string();
        let alt_allel = record
            .alternate_bases()
            .iter()
            .next()
            .and_then(|b| b.ok())
            .map(|b| b.to_string())
            .unwrap_or_else(|| "N".to_string());

        let info = record.info();

        let an: i32 = match info.get(&header, "AN") {
            Some(Ok(Some(Value::Integer(v)))) if v >= 0 => v as i32,
            _ => 0,
        };

        /*let ac: i32 = match info.get(&header, "AC") {
            Some(Ok(Some(Value::Array(Array::Integer(values))))) => values
                .iter()
                .next()
                .and_then(|v| v.ok().flatten())
                .unwrap_or(0)
                as i32,
            _ => 0,
        };*/
        
        let ac: i32 = match info.get(&header, "AC") {
            Some(Ok(Some(Value::Integer(v)))) => v as i32,
            Some(Ok(Some(Value::Array(Array::Integer(values))))) => values
                .iter()
                .next()
                .and_then(|v| v.ok().flatten())
                .unwrap_or(0) as i32,
                
            _ => 0,
        };

        let af: f32 = match info.get(&header, "AF") {
            Some(Ok(Some(Value::Array(Array::Float(values))))) => {
                let mut iter = values.iter();
                if let Some(Ok(Some(v))) = iter.next() {
                    v as f32
                } else {
                    0.0
                }
            }
            _ => 0.0,
        };

        let polarized = parsimony_ancestral(
            &header,
            &record,
            &outgroup_alleles,
            ac,
            af,
            an,
            &ref_allel,
            &alt_allel,
        )?;

        match polarized {
            Some(modified) => {
                writer.write_variant_record(&header, &modified)?;
                flipped_count += 1;
            }
            None => writer.write_record(&header, &record)?,
        }
        written_count += 1;

        write_positions_row(
            &mut *positions_writer,
            &contig,
            pos1,
            block,
            col_idx,
            species_count,
        )?;
    }

    let query_time = total_time.elapsed();
    info!("Time to polarize variants: {:?}", query_time);
    report_polarization(flipped_count, written_count, total_variants);
    Ok(())
}

fn run_model_based(
    maf_path: &Path,
    vcf_path: &Path,
    vcf_out: &Path,
    positions_out: &Path,
    species_names: &[String],
    species_index: &HashMap<String, usize>,
    outgroup_names: &[String],
    outgroup_index: &HashMap<String, usize>,
    method: PolarizationMethod,
    nrandom: usize,
    total_variants: usize,
    report_every: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let n_outgroup = outgroup_names.len();

    let mut positions_writer = open_writer(positions_out)?;
    write_positions_header(&mut *positions_writer, species_names)?;

    let mut contig_rank: HashMap<String, usize> = HashMap::new();
    let mut maf_cursor = MafCursor::new(maf_path, outgroup_index.clone(), species_index.clone())?;

    let mut collected = CollectedData::default();
    let mut positions_to_polarize: HashSet<PosKey> = HashSet::new();

    let est_sfs_out = derived_path(vcf_path, "est_sfs.txt");
    let mut est_writer = BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&est_sfs_out)?,
    );

    let mut reader = vcf::io::reader::Builder::default().build_from_path(vcf_path)?;
    let header = reader.read_header()?;

    let mut processed: usize = 0;
    let total_time = Instant::now();
    let species_count = species_names.len();

    info!("Collecting variants 0/{}", total_variants);
    for row in reader.records() {
        let record = row?;
        processed += 1;
        if processed % report_every == 0 || processed == total_variants {
            info!("Collecting variants {}/{}", processed, total_variants);
        }

        let contig = record.reference_sequence_name().to_string();
        ensure_contig_rank(&mut contig_rank, &contig);

        let pos1: u32 = match record.variant_start() {
            Some(Ok(p)) => p.get() as u32,
            _ => continue,
        };
        let pos0 = pos1.saturating_sub(1);

        let block = match maf_cursor.block_for(&contig, pos0, &contig_rank)? {
            Some(b) => b,
            None => continue,
        };

        if block.seqs.len() < outgroup_index.len() {
            continue;
        }

        let (col_idx, out_bases) = match outgroup_bases_for_variant(block, pos0) {
            Some(v) => v,
            None => continue,
        };

        if out_bases.iter().filter(|b| **b == Base::N).count() > 1 {
            continue;
        }

        let info = record.info();
        let an: i32 = match info.get(&header, "AN") {
            Some(Ok(Some(Value::Integer(v)))) if v >= 0 => v as i32,
            _ => 0,
        };
        /*let ac: i32 = match info.get(&header, "AC") {
            Some(Ok(Some(Value::Array(Array::Integer(values))))) => values
                .iter()
                .next()
                .and_then(|v| v.ok().flatten())
                .unwrap_or(0)
                as i32,
            _ => 0,
        };*/
        let ac: i32 = match info.get(&header, "AC") {
            Some(Ok(Some(Value::Integer(v)))) => v as i32,
            Some(Ok(Some(Value::Array(Array::Integer(values))))) => values
                .iter()
                .next()
                .and_then(|v| v.ok().flatten())
                .unwrap_or(0) as i32,
                
            _ => 0,
        };
        let _af: f32 = match info.get(&header, "AF") {
            Some(Ok(Some(Value::Array(Array::Float(values))))) => {
                let mut iter = values.iter();
                if let Some(Ok(Some(v))) = iter.next() {
                    v as f32
                } else {
                    0.0
                }
            }
            _ => 0.0,
        };
        
        //println!("{:?} {:?} {:?}", ac,_af,an);

        let ref_allel = record.reference_bases().to_string();
        let alt_allel = record
            .alternate_bases()
            .iter()
            .next()
            .and_then(|b| b.ok())
            .map(|b| b.to_string())
            .unwrap_or_else(|| "N".to_string());

        let ref_base = Base::from_str(&ref_allel);
        let alt_base = Base::from_str(&alt_allel);
        if ref_base == Base::N || alt_base == Base::N {
            continue;
        }

        let n_total: u16 = an.try_into().unwrap_or(0);
        let n_alt: u16 = ac.max(0).try_into().unwrap_or(0);
        let n_ref: u16 = n_total.saturating_sub(n_alt);

        let (major, minor, n_major) = if n_ref >= n_alt {
            (ref_base, alt_base, n_ref)
        } else {
            (alt_base, ref_base, n_alt)
        };

        let mut ingroup_counts = [0_i32; 4];
        if let Some(idx) = base_to_index(ref_base) {
            ingroup_counts[idx] += n_ref as i32;
        }
        if let Some(idx) = base_to_index(alt_base) {
            ingroup_counts[idx] += n_alt as i32;
        }

        write!(
            est_writer,
            "{},{},{},{}",
            ingroup_counts[0], ingroup_counts[1], ingroup_counts[2], ingroup_counts[3]
        )?;

        for b in &out_bases {
            let c = base_to_counts(*b);
            write!(est_writer, "\t{},{},{},{}", c[0], c[1], c[2], c[3])?;
        }
        writeln!(est_writer)?;

        let key = ConfigKey {
            major,
            minor,
            n_major,
            n_total,
            outgroups: out_bases.clone(),
        };

        collected.add_site(key);
        positions_to_polarize.insert(PosKey {
            chrom: contig.clone(),
            pos: pos1,
        });

        write_positions_row(
            &mut *positions_writer,
            &contig,
            pos1,
            block,
            col_idx,
            species_count,
        )?;
    }

    let query_time = total_time.elapsed();
    info!(
        "Time to collected {} total sites: {:?} ({} unique configs)",
        &collected.n_sites_total,
        query_time,
        collected.configs.len()
    );

    let total_pol_time = Instant::now();

    let mut variant_configs: Vec<ConfigRow> = Vec::with_capacity(collected.configs.len());
    for (k, m) in collected.configs.iter() {
        let minor_state: i16 = if k.n_major == k.n_total {
            -1
        } else {
            k.minor as i16
        };

        let outgroup_counts: SmallVec<[i8; 8]> = base_to_states(&k.outgroups);
        let major_state = k.major as i16;
        let v = ConfigRow {
            major: major_state,
            minor: minor_state,
            n_major: k.n_major,
            n_total: k.n_total,
            outgroups: outgroup_counts,
            multiplicity: *m,
        };

        variant_configs.push(v);
    }

    let total_time = Instant::now();
    let seed: u64 = rand::rng().random();
    let fit = fit_substitution_model(variant_configs.clone(), n_outgroup, method, nrandom, seed)
        .expect("Stage 1 fit failed");

    let k_str = fit
        .branch_k
        .iter()
        .map(|v| format!("{:.4}", v))
        .collect::<Vec<_>>()
        .join(", ");

    let r6_str = fit
        .r6_rates
        .map(|rates| {
            rates
                .iter()
                .map(|v| format!("{:.4}", v))
                .collect::<Vec<_>>()
                .join(", ")
        })
        .unwrap_or_else(|| "None".to_string());

    info!(
        "Stage 1 fit: method={:?} n_outgroups={}
    NLL   = {}
    k     = {:.4}
    K     = [{}]
    r6    = [{}]",
        fit.method,
        fit.n_outgroups,
        fit.nll,
        fit.kappa.unwrap_or(0.0),
        k_str,
        r6_str
    );
    let query_time = total_time.elapsed();
    info!("Time fit rates: {:?}", query_time);

    let total_time = Instant::now();
    let stage2 = fit_usfs_minor_count(&variant_configs, n_outgroup, &fit, 100, 1e-10);

    let usfs_out = derived_path(vcf_path, "usfs.txt");
    let mut usfs_writer = BufWriter::new(File::create(&usfs_out)?);
    writeln!(usfs_writer, "dac,sfs")?;
    for (i, s) in stage2.expected_derived_counts.iter().enumerate() {
        writeln!(usfs_writer, "{:?},{:?}", i, s)?;
    }
    let query_time = total_time.elapsed();
    info!("Time to estimate uSFS: {:?}", query_time);

    let p_anc_out = derived_path(vcf_path, "p_anc.txt");
    let mut p_anc_writer = BufWriter::new(File::create(&p_anc_out)?);
    writeln!(p_anc_writer, "0 sites {}", positions_to_polarize.len())?;
    writeln!(p_anc_writer, "0 model {:?}", fit.method)?;
    writeln!(p_anc_writer, "0 ML {}", -fit.nll)?;
    writeln!(p_anc_writer, "0 Rates: {:?}", fit.branch_k)?;
    if let Some(k) = fit.kappa {
        writeln!(p_anc_writer, "0 kappa {}", k)?;
    }
    writeln!(p_anc_writer, "0 Chrom Pos P-major-ancestral P-trees[...]")?;

    let mut reader = vcf::io::reader::Builder::default().build_from_path(vcf_path)?;
    let header = reader.read_header()?;

    let mut writer = vcf::io::writer::Builder::default().build_from_path(vcf_out)?;
    writer.write_header(&header)?;

    let mut maf_cursor = MafCursor::new(maf_path, outgroup_index.clone(), species_index.clone())?;
    let mut contig_rank: HashMap<String, usize> = HashMap::new();

    let mut processed: usize = 1;
    let mut flipped_count: usize = 0;
    let mut written_count: usize = 0;
    let total_to_polarize = positions_to_polarize.len();

    info!("Polarizing variants 0/{}", total_to_polarize);

    for row in reader.records() {
        let record = row?;

        let contig = record.reference_sequence_name().to_string();
        let pos1: u32 = match record.variant_start() {
            Some(Ok(p)) => p.get() as u32,
            _ => continue,
        };

        let key = PosKey {
            chrom: contig.clone(),
            pos: pos1,
        };
        if !positions_to_polarize.contains(&key) {
            continue;
        }

        processed += 1;
        if processed % report_every == 0 || processed == total_to_polarize {
            info!("Polarizing variants {}/{}", processed, total_to_polarize);
        }

        ensure_contig_rank(&mut contig_rank, &contig);
        let pos0 = pos1.saturating_sub(1);

        let block = match maf_cursor.block_for(&contig, pos0, &contig_rank)? {
            Some(b) => b,
            None => continue,
        };

        if block.seqs.len() < outgroup_index.len() {
            continue;
        }

        let (_col_idx, out_bases) = match outgroup_bases_for_variant(block, pos0) {
            Some(v) => v,
            None => continue,
        };

        let info = record.info();
        let an: i32 = match info.get(&header, "AN") {
            Some(Ok(Some(Value::Integer(v)))) if v >= 0 => v as i32,
            _ => 0,
        };
        let ac: i32 = match info.get(&header, "AC") {
            Some(Ok(Some(Value::Array(Array::Integer(values))))) => values
                .iter()
                .next()
                .and_then(|v| v.ok().flatten())
                .unwrap_or(0)
                as i32,
            _ => 0,
        };
        let af: f32 = match info.get(&header, "AF") {
            Some(Ok(Some(Value::Array(Array::Float(values))))) => {
                let mut iter = values.iter();
                if let Some(Ok(Some(v))) = iter.next() {
                    v as f32
                } else {
                    0.0
                }
            }
            _ => 0.0,
        };

        let ref_allel = record.reference_bases().to_string();
        let alt_allel = record
            .alternate_bases()
            .iter()
            .next()
            .and_then(|b| b.ok())
            .map(|b| b.to_string())
            .unwrap_or_else(|| "N".to_string());

        let est_result = posterior_ancestral_per_site(
            &header, &record, ac, af, an, &ref_allel, &alt_allel, out_bases, &fit, &stage2,
        )?;

        write!(
            p_anc_writer,
            "{} {} {} {:.6}",
            processed, contig, pos1, est_result.p_major_anc
        )?;
        for val in est_result.pt {
            write!(p_anc_writer, " {:.6}", val)?;
        }
        writeln!(p_anc_writer)?;

        match est_result.modified_record {
            Some(modified) => {
                writer.write_variant_record(&header, &modified)?;
                flipped_count += 1;
            }
            None => writer.write_record(&header, &record)?,
        }
        written_count += 1;
    }

    let query_time = total_pol_time.elapsed();
    info!("Time to polarize variants: {:?}", query_time);
    report_polarization(flipped_count, written_count, total_variants);

    Ok(())
}

struct BlockRecord {
    contig: String,
    start: u64,
    order: u64,
    text: String,
}

struct HeapItem {
    record: BlockRecord,
    rank: usize,
    file_idx: usize,
}

impl HeapItem {
    fn cmp_key(&self, other: &Self) -> std::cmp::Ordering {
        if self.rank != other.rank {
            return self.rank.cmp(&other.rank);
        }
        if self.rank == usize::MAX && self.record.contig != other.record.contig {
            return self.record.contig.cmp(&other.record.contig);
        }
        if self.record.start != other.record.start {
            return self.record.start.cmp(&other.record.start);
        }
        self.record.order.cmp(&other.record.order)
    }
}

impl PartialEq for HeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.cmp_key(other) == std::cmp::Ordering::Equal
    }
}

impl Eq for HeapItem {}

impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // reverse for min-heap
        other.cmp_key(self)
    }
}

fn run_sort_maf(args: SortMafArgs) -> Result<(), Box<dyn std::error::Error>> {
    let contig_order = match &args.contig_order {
        Some(path) => read_contig_order(path)?,
        None => HashMap::new(),
    };

    let tmp_dir = args
        .tmp_dir
        .unwrap_or_else(|| std::env::temp_dir().join("polarize_maf_sort"));
    create_dir_all(&tmp_dir)?;

    let (header_lines, chunk_files) =
        build_sorted_chunks(&args.input, &contig_order, args.chunk_bytes, &tmp_dir)?;

    let mut writer = open_writer(&args.output)?;
    for line in header_lines {
        writer.write_all(line.as_bytes())?;
    }

    merge_chunks(&chunk_files, &contig_order, &mut *writer)?;

    for path in chunk_files {
        let _ = remove_file(path);
    }

    Ok(())
}

fn read_contig_order(path: &Path) -> Result<HashMap<String, usize>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut map: HashMap<String, usize> = HashMap::new();

    for line in reader.lines() {
        let line = line?;
        let contig = line.split_whitespace().next().unwrap_or("");
        if contig.is_empty() {
            continue;
        }
        if !map.contains_key(contig) {
            let next = map.len();
            map.insert(contig.to_string(), next);
        }
    }

    Ok(map)
}

fn is_gzipped(path: &Path) -> io::Result<bool> {
    let mut file = File::open(path)?;
    let mut magic = [0_u8; 2];
    let n = file.read(&mut magic)?;
    Ok(n == 2 && magic == [0x1f, 0x8b])
}

fn open_maf_reader(path: &Path) -> io::Result<Box<dyn BufRead>> {
    let file = File::open(path)?;
    if is_gzipped(path)? {
        Ok(Box::new(BufReader::new(flate2::read::MultiGzDecoder::new(
            file,
        ))))
    } else {
        Ok(Box::new(BufReader::new(file)))
    }
}

fn build_sorted_chunks(
    input: &Path,
    contig_order: &HashMap<String, usize>,
    chunk_bytes: usize,
    tmp_dir: &Path,
) -> Result<(Vec<String>, Vec<PathBuf>), Box<dyn std::error::Error>> {
    let mut reader = open_maf_reader(input)?;
    let mut header_lines: Vec<String> = Vec::new();
    let mut block_lines: Vec<String> = Vec::new();
    let mut blocks: Vec<BlockRecord> = Vec::new();
    let mut chunk_files: Vec<PathBuf> = Vec::new();
    let mut bytes_in_chunk: usize = 0;
    let mut order: u64 = 0;
    let mut saw_block = false;

    let mut line = String::new();
    loop {
        line.clear();
        let bytes = reader.read_line(&mut line)?;
        if bytes == 0 {
            break;
        }

        let trimmed = line.trim_start();
        let is_blank = trimmed.trim_end().is_empty();
        let is_block_start = trimmed.starts_with('a')
            && trimmed
                .as_bytes()
                .get(1)
                .map(|b| b.is_ascii_whitespace())
                .unwrap_or(false);

        if !saw_block {
            if is_block_start {
                saw_block = true;
                block_lines.push(line.clone());
            } else {
                header_lines.push(line.clone());
            }
            continue;
        }

        if is_block_start && !block_lines.is_empty() {
            let record = block_from_lines(&block_lines, order);
            bytes_in_chunk += record.text.len();
            blocks.push(record);
            order += 1;
            block_lines.clear();
            block_lines.push(line.clone());
        } else if is_blank {
            if !block_lines.is_empty() {
                block_lines.push(line.clone());
                let record = block_from_lines(&block_lines, order);
                bytes_in_chunk += record.text.len();
                blocks.push(record);
                order += 1;
                block_lines.clear();
            }
        } else {
            block_lines.push(line.clone());
        }

        if bytes_in_chunk >= chunk_bytes && !blocks.is_empty() {
            let chunk_path = write_chunk(&mut blocks, contig_order, tmp_dir, chunk_files.len())?;
            chunk_files.push(chunk_path);
            bytes_in_chunk = 0;
        }
    }

    if !block_lines.is_empty() {
        let record = block_from_lines(&block_lines, order);
        blocks.push(record);
    }

    if !blocks.is_empty() {
        let chunk_path = write_chunk(&mut blocks, contig_order, tmp_dir, chunk_files.len())?;
        chunk_files.push(chunk_path);
    }

    Ok((header_lines, chunk_files))
}

fn block_from_lines(lines: &[String], order: u64) -> BlockRecord {
    let mut contig = String::new();
    let mut start: u64 = 0;

    for line in lines {
        let trimmed = line.trim_start();
        if trimmed.starts_with('s')
            && trimmed
                .as_bytes()
                .get(1)
                .map(|b| b.is_ascii_whitespace())
                .unwrap_or(false)
        {
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if parts.len() >= 3 {
                let src = parts[1];
                start = parts[2].parse().unwrap_or(0);
                contig = src.splitn(2, '.').nth(1).unwrap_or(src).to_string();
            }
            break;
        }
    }

    if contig.is_empty() && !lines.iter().all(|l| l.trim().is_empty() || l.trim().starts_with('#')) {
        warn!("MAF block missing reference contig; placing at end");
    }

    BlockRecord {
        contig,
        start,
        order,
        text: lines.concat(),
    }
}

fn write_chunk(
    blocks: &mut Vec<BlockRecord>,
    contig_order: &HashMap<String, usize>,
    tmp_dir: &Path,
    chunk_idx: usize,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    blocks.sort_by(|a, b| compare_records(a, b, contig_order));

    let pid = std::process::id();
    let path = tmp_dir.join(format!("maf_sort_chunk_{}_{}.bin", pid, chunk_idx));
    let mut writer = BufWriter::new(File::create(&path)?);

    for block in blocks.drain(..) {
        write_u32(&mut writer, block.contig.len() as u32)?;
        writer.write_all(block.contig.as_bytes())?;
        write_u64(&mut writer, block.start)?;
        write_u64(&mut writer, block.order)?;
        write_u64(&mut writer, block.text.len() as u64)?;
        writer.write_all(block.text.as_bytes())?;
    }

    Ok(path)
}

fn read_block_record(reader: &mut BufReader<File>) -> io::Result<Option<BlockRecord>> {
    let contig_len = match read_u32(reader) {
        Ok(v) => v as usize,
        Err(err) if err.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(err) => return Err(err),
    };

    let mut contig_buf = vec![0_u8; contig_len];
    reader.read_exact(&mut contig_buf)?;
    let contig = String::from_utf8(contig_buf).unwrap_or_default();

    let start = read_u64(reader)?;
    let order = read_u64(reader)?;
    let text_len = read_u64(reader)? as usize;

    let mut text_buf = vec![0_u8; text_len];
    reader.read_exact(&mut text_buf)?;
    let text = String::from_utf8(text_buf).unwrap_or_default();

    Ok(Some(BlockRecord {
        contig,
        start,
        order,
        text,
    }))
}

fn merge_chunks(
    chunk_files: &[PathBuf],
    contig_order: &HashMap<String, usize>,
    writer: &mut dyn Write,
) -> Result<(), Box<dyn std::error::Error>> {
    if chunk_files.is_empty() {
        return Ok(());
    }

    let mut readers: Vec<BufReader<File>> = Vec::new();
    for path in chunk_files {
        readers.push(BufReader::new(File::open(path)?));
    }

    let mut heap: BinaryHeap<HeapItem> = BinaryHeap::new();
    for (idx, reader) in readers.iter_mut().enumerate() {
        if let Some(record) = read_block_record(reader)? {
            let rank = contig_order
                .get(&record.contig)
                .copied()
                .unwrap_or(usize::MAX);
            heap.push(HeapItem {
                record,
                rank,
                file_idx: idx,
            });
        }
    }

    while let Some(item) = heap.pop() {
        writer.write_all(item.record.text.as_bytes())?;

        let reader = &mut readers[item.file_idx];
        if let Some(next_record) = read_block_record(reader)? {
            let rank = contig_order
                .get(&next_record.contig)
                .copied()
                .unwrap_or(usize::MAX);
            heap.push(HeapItem {
                record: next_record,
                rank,
                file_idx: item.file_idx,
            });
        }
    }

    Ok(())
}

fn compare_records(
    a: &BlockRecord,
    b: &BlockRecord,
    contig_order: &HashMap<String, usize>,
) -> std::cmp::Ordering {
    let a_rank = contig_order.get(&a.contig).copied();
    let b_rank = contig_order.get(&b.contig).copied();

    match (a_rank, b_rank) {
        (Some(ar), Some(br)) => {
            if ar != br {
                return ar.cmp(&br);
            }
        }
        (Some(_), None) => return std::cmp::Ordering::Less,
        (None, Some(_)) => return std::cmp::Ordering::Greater,
        (None, None) => {
            if a.contig != b.contig {
                return a.contig.cmp(&b.contig);
            }
        }
    }

    if a.start != b.start {
        return a.start.cmp(&b.start);
    }

    a.order.cmp(&b.order)
}

fn write_u32(writer: &mut dyn Write, value: u32) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn write_u64(writer: &mut dyn Write, value: u64) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn read_u32(reader: &mut dyn Read) -> io::Result<u32> {
    let mut buf = [0_u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64(reader: &mut dyn Read) -> io::Result<u64> {
    let mut buf = [0_u8; 8];
    reader.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}


fn run_split_maf(args: SplitMafArgs) -> Result<(), Box<dyn std::error::Error>> {
    let mut reader = open_maf_reader(&args.input)?;
    let mut writer = open_writer(&args.output)?;

    let mut block_lines: Vec<String> = Vec::new();
    let mut saw_block = false;
    let mut blocks_written = 0;

    info!("Splitting MAF, filtering for contig: {}", args.contig);

    let mut line = String::new();
    loop {
        line.clear();
        let bytes = reader.read_line(&mut line)?;
        if bytes == 0 {
            break;
        }

        let trimmed = line.trim_start();
        let is_blank = trimmed.trim_end().is_empty();
        let is_block_start = trimmed.starts_with('a')
            && trimmed
                .as_bytes()
                .get(1)
                .map(|b| b.is_ascii_whitespace())
                .unwrap_or(false);

        if !saw_block {
            if is_block_start {
                saw_block = true;
                block_lines.push(line.clone());
            } else {
                // Pass header lines straight to the output
                writer.write_all(line.as_bytes())?;
            }
            continue;
        }

        if is_block_start && !block_lines.is_empty() {
            if process_split_block(&block_lines, &args.contig, &mut *writer)? {
                blocks_written += 1;
            }
            block_lines.clear();
            block_lines.push(line.clone());
        } else if is_blank {
            if !block_lines.is_empty() {
                block_lines.push(line.clone());
                if process_split_block(&block_lines, &args.contig, &mut *writer)? {
                    blocks_written += 1;
                }
                block_lines.clear();
            }
        } else {
            block_lines.push(line.clone());
        }
    }

    // Process the final block if the file didn't end with a blank line
    if !block_lines.is_empty() {
        if process_split_block(&block_lines, &args.contig, &mut *writer)? {
            blocks_written += 1;
        }
    }

    info!("Finished splitting. Wrote {} blocks for contig '{}'", blocks_written, args.contig);
    Ok(())
}

/// Helper function to check a block's contig and write it if it matches
fn process_split_block(
    lines: &[String],
    target_contig: &str,
    writer: &mut dyn Write,
) -> io::Result<bool> {
    // Reusing the existing block_from_lines function
    let record = block_from_lines(lines, 0);
    if record.contig == target_contig {
        writer.write_all(record.text.as_bytes())?;
        return Ok(true);
    }
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn make_temp_dir(prefix: &str) -> PathBuf {
        let mut dir = std::env::temp_dir();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        dir.push(format!("{}_{}_{}", prefix, std::process::id(), nanos));
        create_dir_all(&dir).unwrap();
        dir
    }

    fn ref_order_from_maf(path: &Path) -> Vec<(String, u64)> {
        let file = File::open(path).unwrap();
        let reader = BufReader::new(file);
        let mut order = Vec::new();
        let mut in_block = false;
        let mut got_ref = false;

        for line in reader.lines() {
            let line = line.unwrap();
            let trimmed = line.trim();
            if trimmed.is_empty() {
                in_block = false;
                got_ref = false;
                continue;
            }
            if trimmed.starts_with('a') {
                in_block = true;
                got_ref = false;
                continue;
            }
            if in_block && !got_ref && trimmed.starts_with('s') {
                let parts: Vec<&str> = trimmed.split_whitespace().collect();
                if parts.len() >= 3 {
                    let src = parts[1];
                    let start: u64 = parts[2].parse().unwrap();
                    let contig = src.splitn(2, '.').nth(1).unwrap_or(src).to_string();
                    order.push((contig, start));
                    got_ref = true;
                }
            }
        }

        order
    }

    fn sorted_expected(order: &[(String, u64)]) -> Vec<(String, u64)> {
        let mut out = order.to_vec();
        out.sort_by(|a, b| {
            if a.0 != b.0 {
                return a.0.cmp(&b.0);
            }
            a.1.cmp(&b.1)
        });
        out
    }

    fn count_nonref_contig_mismatch(path: &Path) -> usize {
        let file = File::open(path).unwrap();
        let reader = BufReader::new(file);
        let mut in_block = false;
        let mut ref_contig: Option<String> = None;
        let mut mismatches = 0;

        for line in reader.lines() {
            let line = line.unwrap();
            let trimmed = line.trim();
            if trimmed.is_empty() {
                in_block = false;
                ref_contig = None;
                continue;
            }
            if trimmed.starts_with('a') {
                in_block = true;
                ref_contig = None;
                continue;
            }
            if in_block && trimmed.starts_with('s') {
                let parts: Vec<&str> = trimmed.split_whitespace().collect();
                if parts.len() < 2 {
                    continue;
                }
                let src = parts[1];
                let contig = src.splitn(2, '.').nth(1).unwrap_or(src).to_string();
                if ref_contig.is_none() {
                    ref_contig = Some(contig);
                } else if let Some(ref rc) = ref_contig {
                    if &contig != rc {
                        mismatches += 1;
                    }
                }
            }
        }

        mismatches
    }

    fn check_sorted_and_counts(input: &Path, output: &Path) {
        let input_order = ref_order_from_maf(input);
        let output_order = ref_order_from_maf(output);

        assert_eq!(input_order.len(), 100, "expected 100 blocks in input");
        assert_eq!(output_order.len(), 100, "expected 100 blocks in output");

        let expected = sorted_expected(&input_order);
        assert_eq!(output_order, expected, "output blocks not sorted correctly");
    }

    #[test]
    fn sort_maf_single_contig_by_position() {
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let input = manifest_dir.join("tests/data/single_contig_unsorted.maf");
        let temp_dir = make_temp_dir("maf_sort_single");
        let chunk_dir = temp_dir.join("chunks");
        create_dir_all(&chunk_dir).unwrap();
        let output = temp_dir.join("sorted.maf");

        let args = SortMafArgs {
            input: input.clone(),
            output: output.clone(),
            contig_order: None,
            chunk_bytes: 2048,
            tmp_dir: Some(chunk_dir),
        };

        run_sort_maf(args).unwrap();
        check_sorted_and_counts(&input, &output);
    }

    #[test]
    fn sort_maf_multi_contig_then_position() {
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let input = manifest_dir.join("tests/data/multi_contig_unsorted.maf");
        let temp_dir = make_temp_dir("maf_sort_multi");
        let chunk_dir = temp_dir.join("chunks");
        create_dir_all(&chunk_dir).unwrap();
        let output = temp_dir.join("sorted.maf");

        let args = SortMafArgs {
            input: input.clone(),
            output: output.clone(),
            contig_order: None,
            chunk_bytes: 2048,
            tmp_dir: Some(chunk_dir),
        };

        run_sort_maf(args).unwrap();
        check_sorted_and_counts(&input, &output);

        let mismatches = count_nonref_contig_mismatch(&input);
        assert!(
            mismatches > 0,
            "expected at least one non-reference contig mismatch in input"
        );
    }
}


// 1. Wrap the Polarize logic
/// Polarize a VCF using a contig/pos-sorted MAF via streaming merge-join.
///
/// :param str maf: Input MAF file (sorted by contig/pos).
/// :param str vcf: Input VCF file (sorted by contig/pos).
/// :param str outgroup: Comma-separated list of outgroups (e.g., "panTro4,gorGor6").
/// :param str method: Method to polarize ("parsimony" or "kimura").
/// :param int nrandom: Number of random starts for model fitting.
/// :param str output: Output VCF path.
#[pyfunction]
#[pyo3(name = "polarize")]
#[pyo3(signature = (maf, vcf, outgroup, method=None, nrandom=10, output=None))]
pub fn run_polarize_py(
    maf: String,
    vcf: String,
    outgroup: String,
    method: Option<String>,
    nrandom: usize,
    output: Option<String>,
) -> PyResult<()> {
    let args = PolarizeArgs {
        maf: PathBuf::from(maf),
        vcf: PathBuf::from(vcf),
        outgroup,
        method: method.map(|m| match m.as_str() {
            "parsimony" => PolarizationMethod::Parsimony,
            "jc" => PolarizationMethod::JC,
            "kimura" => PolarizationMethod::Kimura,
            "r6" => PolarizationMethod::R6,
            _ => PolarizationMethod::Kimura, // Default
        }),
        nrandom,
        output: output.map(PathBuf::from),
    };

    run_polarize(args).map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

// 2. Wrap the MAF Sort logic
/// Sort a MAF by reference contig and position.
///
/// :param str input: Input MAF file to sort.
/// :param str output: Output sorted MAF file.
/// :param str contig_order: Optional path to contig order file or .fai.
/// :param int chunk_bytes: Max bytes in memory before spilling to disk.
/// :param str tmp_dir: Temporary directory for sort chunks.
#[pyfunction]
#[pyo3(name = "maf_sort")]
#[pyo3(signature = (input, output, contig_order=None, chunk_bytes=268435456, tmp_dir=None))]
pub fn maf_sort_py(
    input: String,
    output: String,
    contig_order: Option<String>,
    chunk_bytes: usize,
    tmp_dir: Option<String>,
) -> PyResult<()> {
    let args = SortMafArgs {
        input: PathBuf::from(input),
        output: PathBuf::from(output),
        contig_order: contig_order.map(PathBuf::from),
        chunk_bytes,
        tmp_dir: tmp_dir.map(PathBuf::from),
    };

    run_sort_maf(args).map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

// 3. Split MAF by contig
/// Split a MAF by reference contig.
///
/// :param str input: Path to input MAF file (can be gzipped).
/// :param str output: Path to output MAF file (can be gzipped).
/// :param str contig: The reference contig name to extract.
#[pyfunction]
#[pyo3(name = "maf_split")]
#[pyo3(signature = (input, output, contig))]
pub fn maf_split_py(
    input: String,
    output: String,
    contig: String,
) -> PyResult<()> {
    let args = SplitMafArgs {
        input: PathBuf::from(input),
        output: PathBuf::from(output),
        contig,
    };

    run_split_maf(args).map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

#[pyfunction]
pub fn init_logging() {
    // This initializes the tracing ecosystem
    // It will look for RUST_LOG env var, or default to "info"
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));

        tracing_subscriber::registry()
                .with(
                    fmt::layer()
                        .with_writer(std::io::stderr)
                        .without_time()
                        .with_target(false)
                        .with_level(true)
                )
        .with(filter)
        .try_init()
        .ok();

    //tracing::info!("Tracing initialized");
}

#[pymodule]
fn polarize(m: &Bound<'_, PyModule>) -> PyResult<()> {

    init_logging();
    m.add_function(wrap_pyfunction!(init_logging, m)?)?;
    m.add_function(wrap_pyfunction!(maf_split_py, m)?)?;
    m.add_function(wrap_pyfunction!(maf_sort_py, m)?)?;
    m.add_function(wrap_pyfunction!(run_polarize_py, m)?)?;
    Ok(())

}