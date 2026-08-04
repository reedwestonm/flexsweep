import math
import os
import click


def parse_float_list(value):
    """Parse a comma-separated list of floats."""
    if value:
        try:
            return [int(x) if x.isdigit() else float(x) for x in value.split(",")]
        except ValueError:
            raise click.BadParameter("Must be a comma-separated list of floats")
    return None


@click.group()
def cli():
    """CLI for Simulator and CNN."""
    pass


@cli.command()
@click.option("--sample_size", type=int, required=True, help="Number of haplotypes")
@click.option(
    "--mutation_rate",
    type=str,
    required=False,
    default="5e-9,2e-8",
    help=(
        "Mutation rate specification. "
        "Please input:\n"
        "  - Two comma-separated values: lower,upper (uniform distribution bounds):"
        "  - Three values: min, max and mean of an exponential distribution."
        "Example: '5e-9,2e-8' or '5e-9,2e-8,1e-8'"
    ),
)
@click.option(
    "--recombination_rate",
    type=str,
    required=False,
    default="1e-10,1e-7,1e-8",
    help=(
        "Recombination rate specification. "
        "Please input:"
        "  - Two comma-separated values: lower,upper (uniform distribution bounds):"
        "  - Three values: min, max and mean of an exponential distribution."
        "Example: '1e-9,4e-8' or '1e-10,1e-7,1e-8'"
    ),
)
@click.option(
    "--locus_length",
    type=int,
    required=False,
    default=int(1.2e6),
    help="Length of the simulated locus in base pairs.",
)
@click.option(
    "--demes",
    type=str,
    required=True,
    help="Path to the demes YAML file describing demography.",
)
@click.option(
    "--output_folder",
    type=str,
    required=True,
    help="Directory where simulation outputs will be saved.",
)
@click.option(
    "--time",
    type=str,
    default="0,5000",
    help=(
        "Adaptive mutation time range in generations. "
        "Two comma-separated values: start,end. "
        "Default: '0,5000'"
    ),
)
@click.option(
    "--s",
    type=str,
    default="0.01,0.05",
    help=(
        "Selection coefficients."
        "Two comma-separated values: min,max "
        "Default: '0.01,0.05'"
    ),
)
@click.option(
    "--saf",
    type=str,
    default="0.0,0.1",
    help=("Start allel frequency.Two comma-separated values: min,max Default: '0,0.1'"),
)
@click.option(
    "--eaf",
    type=str,
    default="0.5,1",
    help=("End allel frequency.Two comma-separated values: min,max Default: '0.5,1'"),
)
@click.option(
    "--num_simulations",
    type=int,
    default=int(2.5e5),
    help="Number of neutral and sweep simulations to generate. Default: 250000.",
)
@click.option(
    "--nthreads",
    type=int,
    default=1,
    help="Number of threads for parallelization. Default: 1.",
)
@click.option(
    "--discoal_path",
    type=str,
    default=None,
    help=(
        "Path to the discoal executable. If not provided, using pre-compiled flexsweep.DISCOAL."
    ),
)
def simulator(
    sample_size,
    mutation_rate,
    recombination_rate,
    locus_length,
    demes,
    output_folder,
    discoal_path,
    num_simulations,
    time,
    s,
    saf,
    eaf,
    nthreads,
):
    """
    Run the discoal simulator with user-specified parameters.

    flexsweep.Simulator class, parsing mutation and
    recombination rate specifications from the command line, and dispatches
    neutral and sweep simulations to discoal.

    \b
    Example usage:
        flexsweep simulate --sample_size 20 --demes model.yaml --output_folder ./sims --nthreads 24

    """
    import flexsweep as fs

    # If not provided explicitly, use default path from flexsweep
    if discoal_path is None:
        discoal_path = fs.DISCOAL

    mutation_rate_list = parse_float_list(mutation_rate)
    recombination_rate_list = parse_float_list(recombination_rate)

    # Build mutation rate distribution spec
    if len(mutation_rate_list) == 2:
        mu_rate = {
            "dist": "uniform",
            "min": mutation_rate_list[0],
            "max": mutation_rate_list[1],
        }
    elif len(mutation_rate_list) == 3:
        mu_rate = {
            "dist": "exponential",
            "min": mutation_rate_list[0],
            "max": mutation_rate_list[1],
            "mean": mutation_rate_list[2],
        }

    if len(recombination_rate_list) == 1:
        rho_rate = {
            "dist": "exponential",
            "mean": recombination_rate_list[0],
        }
    elif len(recombination_rate_list) == 2:
        rho_rate = {
            "dist": "uniform",
            "lower": recombination_rate_list[0],
            "upper": recombination_rate_list[1],
        }
    elif len(recombination_rate_list) == 3:
        rho_rate = {
            "dist": "exponential",
            "min": recombination_rate_list[0],
            "max": recombination_rate_list[1],
            "mean": recombination_rate_list[2],
        }

    # Parse time range (not directly used in this wrapper, passed to Simulator internally)
    time_list = parse_float_list(time)
    s_list = parse_float_list(s)
    saf_list = parse_float_list(saf)
    eaf_list = parse_float_list(eaf)

    # Instantiate Simulator and run simulations
    simulator = fs.Simulator(
        sample_size=sample_size,
        mutation_rate=mu_rate,
        recombination_rate=rho_rate,
        locus_length=locus_length,
        demes=demes,
        output_folder=output_folder,
        discoal_path=fs.DISCOAL,
        num_simulations=num_simulations,
        nthreads=nthreads,
    )

    simulator.time = time_list
    simulator.s = s_list
    simulator.f_i = saf_list
    simulator.f_t = eaf_list
    simulator.create_params()
    simulator.simulate_batch()


# ─────────────────────────────────────────────────────────────────────────────
# WINDOWING SCHEMES for --windows / --centers (locus_length = 1.2e6, sweep @ 600 kb)
# --windows = comma-list of window sizes (bp); --centers = comma-list of center
# positions (bp). Omit --centers to use the uniform grid derived from --step.
# Same flags apply to `fvs-discoal` (sims) and `fvs-vcf` (centers are sim-relative,
# remapped per region). The five schemes used in the s/t bake-off:
#
#   base          --windows 100000
#                 (uniform centers from --step 100000 -> 50k..1.15M)              12 x 1
#
#   ms            --windows 20000,50000,100000,200000,500000
#                 (uniform centers from --step 100000)                            12 x 5
#
#   log_scale     --windows 2000,4000,8000,16000,32000,64000,128000,256000,512000
#                 --centers 10000,30000,...,1190000  (range(10000,1200000,20000)) 60 x 9
#
#   log_distance  --windows 4000,16000,64000,256000
#                 --centers 280000,440000,520000,560000,580000,600000,620000,
#                           640000,680000,760000,920000  (log-dist radial @ 600k) 11 x 4
#
#   radial        --windows 4000,8000,16000,32000,64000,128000,256000,512000
#                 --centers 600000  (single sweep center)                          1 x 8
#
#   caldas        Caldas et al. 2022 geometry (biorxiv 2022.07.19.500702), the
#                 d x d subwindow cube with d = 14 rescaled to L = 1.2 Mb.
#                 NOT a single call: summary_statistics applies ONE --centers to
#                 ALL --windows (rectangular product, fv.py:1934), but Caldas
#                 centers are SIZE-DEPENDENT (spacing = size/2, fanned on sweep).
#                 => run ONCE PER size s_i, each with its own 21-center fan, then
#                 hstack the 21 parquets on the index cols. Driver: scripts/caldas_geometry.
#                   sizes (21, geometric lmin=1e3 -> L/11=109091, ratio 1.2644):
#                     1000,1264,1599,2021,2556,3232,4086,5167,6533,8260,10445,
#                     13206,16698,21114,26696,33755,42681,53966,68235,86278,109091
#                   per size s: --centers = [600000 + (j-10)*(s/2) for j in 0..20]
#                     (21 windows overlapping neighbours by half, fanned @ 600k)  21 x 21
# ─────────────────────────────────────────────────────────────────────────────
@cli.command()
@click.option(
    "--simulations_path",
    type=str,
    required=True,
    help="Directory containing neutral and sweeps discoal simulations.",
)
@click.option(
    "--stats",
    type=str,
    required=False,
    default=None,
    help="Comma-separated list of summary statistics to compute, e.g:dind,fay_wu_h,isafe",
)
@click.option(
    "--nthreads",
    type=int,
    required=True,
    default=1,
    help="Number of thread to parallelize.",
)
@click.option(
    "--windows",
    type=str,
    required=False,
    default="100000",
    help="Comma-separated window size(s) in bp, e.g. 20000,50000,100000,200000,500000 (ms). See WINDOWING SCHEMES comment above.",
)
@click.option(
    "--step",
    type=float,
    required=False,
    default=int(1e5),
    help="Step size to sliding windows over the simulated regions",
)
@click.option(
    "--centers",
    type=str,
    required=False,
    default=None,
    help="Comma-separated center position(s) in bp. Omit for the uniform grid from --step. See WINDOWING SCHEMES comment above.",
)
@click.option(
    "--locus_length",
    type=float,
    required=False,
    default=int(1.2e6),
    help="Simulated locus size",
)
@click.option(
    "--r_bins",
    type=str,
    required=False,
    default=None,
    help="Recombination bins to normalize simulations",
)
@click.option(
    "--suffix",
    type=str,
    required=False,
    default=None,
    help="Custom suffix name to save feature vectors",
)
@click.option(
    "--save_stats",
    type=bool,
    required=False,
    default=False,
    help="Save raw statistics.",
)
@click.option(
    "--only_normalize",
    type=bool,
    required=False,
    default=False,
    help="Only normalizing raw statistics.",
)
def fvs_discoal(
    simulations_path,
    stats,
    windows,
    step,
    centers,
    locus_length,
    r_bins,
    save_stats,
    suffix,
    nthreads,
    only_normalize,
):
    """
    Estimate summary statistics from discoal simulations and build feature vectors.

    This command processes both neutral and sweep simulations in the given directory,
    computes a panel of summary statistics, and generates two outputs: a Parquet dataframe containing feature vectors, a Pickle dictionary containing neutral expectations and standard deviations (used for normalization during CNN training).
    """
    import flexsweep as fs

    r_bins_list = parse_float_list(r_bins)
    windows_list = parse_float_list(windows)
    center_list = parse_float_list(centers)

    stats_list = stats.split(",") if stats is not None else None

    if only_normalize:
        fs.fv._normalize_raw_stats(
            simulations_path,
            nthreads,
            center_list,
            windows_list,
            step,
            r_bins=r_bins_list,
            suffix=suffix,
            vcf=False,
        )
    else:
        print("Estimating summary statistics")
        fs.fv.summary_statistics(
            simulations_path,
            stats=stats_list,
            vcf=False,
            nthreads=nthreads,
            windows=windows_list,
            step=step,
            centers=center_list,
            locus_length=locus_length,
            recombination_map=None,
            r_bins=r_bins_list,
            min_rate=0.0,
            suffix=suffix,
            func=None,
            save_stats=save_stats,
        )


@cli.command()
@click.option(
    "--vcf_path",
    type=str,
    required=True,
    help="Directory containing vcfs folder with all the VCF files to analyze.",
)
@click.option(
    "--stats",
    type=str,
    required=False,
    default=None,
    help="Comma-separated list of summary statistics to compute, e.g:dind,fay_wu_h,isafe",
)
@click.option(
    "--nthreads",
    type=int,
    required=False,
    default=1,
    help="Number of threads for parallelization",
)
@click.option(
    "--windows",
    type=float,
    required=False,
    default=int(1e5),
    help="Window size to sliding windows over the selected region",
)
@click.option(
    "--step",
    type=int,
    required=False,
    default=int(1e5),
    help="Step size to sliding windows over the selected region",
)
@click.option(
    "--step_vcf",
    type=float,
    required=False,
    default=int(1e4),
    help="Step size to sliding windows over the VCF",
)
@click.option(
    "--locus_length",
    type=float,
    required=False,
    default=int(1.2e6),
    help="Windows size to sliding windows over the VCF",
)
@click.option(
    "--recombination_map",
    type=str,
    default=None,
    required=False,
    help="Recombination map. Decode CSV format: Chr,Begin,End,cMperMb,cM",
)
@click.option(
    "--min_rate",
    type=float,
    required=False,
    default=0.01,
    help="Minimun recombination rate simulated",
)
@click.option(
    "--r_bins",
    type=str,
    required=False,
    default=None,
    help="Recombination bins to normalize simulations",
)
@click.option(
    "--suffix",
    type=str,
    required=True,
    default=None,
    help="Custom suffix name to save feature vectors",
)
@click.option(
    "--save_stats",
    type=bool,
    required=False,
    default=False,
    help="Save raw statistics.",
)
@click.option(
    "--only_normalize",
    type=bool,
    required=False,
    default=False,
    help="Only normalizing previous raw statistics.",
)
def fvs_vcf(
    vcf_path,
    stats,
    windows,
    step,
    step_vcf,
    locus_length,
    recombination_map,
    min_rate,
    r_bins,
    save_stats,
    suffix,
    nthreads,
    only_normalize,
):
    """
    Estimate summary statistics from VCF files and build feature vectors.

    This command parses VCF files in the given directory, computes summary statistics
    per genomic window, and writes feature vectors suitable as CNN input.

    \b
    Example usage:
        # Run summary statistics from VCFs using 8 threads, no recombination map
        flexsweep fvs-vcf --vcf_path ./data --nthreads 8
    \b
        # Run with a recombination map
        flexsweep fvs-vcf --vcf_path ./data --nthreads 8 --recombination_map recomb_map.csv

    Notes: VCF files must be bgzipped and tabix-indexed.
    """
    import flexsweep as fs

    r_bins_list = parse_float_list(r_bins)

    stats_list = stats.split(",") if stats is not None else None
    if only_normalize:
        fs.fv.normalize_raw_stats(
            vcf_path,
            nthreads,
            center_list,
            [windows],
            step,
            r_bins=r_bins_list,
            suffix=suffix,
            vcf=True,
        )
    else:
        fs.summary_statistics(
            vcf_path,
            vcf=True,
            stats=stats_list,
            nthreads=nthreads,
            windows=[windows],
            step=step,
            step_vcf=step_vcf,
            locus_length=locus_length,
            recombination_map=recombination_map,
            r_bins=r_bins_list,
            min_rate=min_rate,
            suffix=suffix,
            func=None,
            save_stats=save_stats,
        )


@cli.command()
@click.option(
    "--vcf_path",
    type=str,
    required=True,
    help="Directory containing vcfs folder with all the VCF files to analyze.",
)
@click.option(
    "--recombination_map",
    type=str,
    default=None,
    required=False,
    help="Recombination map. Decode CSV format: Chr,Begin,End,cMperMb,cM",
)
@click.option(
    "--bins", type=int, default=10, required=False, help="Recombinations bins"
)
@click.option(
    "--window_size",
    type=int,
    default=int(1.2e6),
    required=False,
    help="Genome windows size",
)
@click.option(
    "--step", type=int, default=int(1e4), required=False, help="Step to slide windows"
)
@click.option(
    "--min_rate",
    type=float,
    default=0.01,
    required=False,
    help="Minimun recombination rate (cM/Mb)",
)
def recombination_bins(vcf_path, recombination_map, bins, window_size, step, min_rate):
    """
    Output recombination bins from empirical recombination map in decode format.

    \b
    Example usage:
        flexsweep recombination-bins --vcf_path ./data --recombination_map decode.tsv

    """
    import glob
    import os

    from joblib import Parallel, delayed

    import flexsweep as fs
    from flexsweep import pl
    from flexsweep.fv import get_cm

    def read_regions(vcf_path):
        fs_data = fs.Data(vcf_path, window_size=window_size, step=step, nthreads=1)
        vcf_dict = fs_data.read_vcf()

        regions = (
            pl.DataFrame(vcf_dict["region"])
            .with_columns(
                pl.col("column_0")
                .str.extract_groups(r"^(?P<chr>[^:]+):(?P<start>\d+)-(?P<end>\d+)$")
                .alias("g")
            )
            .unnest("g")
            .with_columns(
                pl.col("start").cast(pl.Int64),
                pl.col("end").cast(pl.Int64),
            )
            .drop("column_0")
        )
        tmp_r = get_cm(
            df_recombination_map.filter(
                pl.col("chr") == regions["chr"].unique().item()
            ),
            regions.select("start", "end").to_numpy(),
            cm_mb=True,
        )
        return tmp_r

    # fvs_file = {}
    # regions = {}

    vcf_files = sorted(
        glob.glob(os.path.join(vcf_path, "*vcf.gz"))
        + glob.glob(os.path.join(vcf_path, "*bcf.gz"))
    )

    if not vcf_files:
        raise FileNotFoundError(f"No VCF/BCF files found in directory: {vcf_path}")

    df_recombination_map = pl.read_csv(
        recombination_map,
        separator="\t",
        comment_prefix="#",
        schema=pl.Schema(
            [
                ("chr", pl.String),
                ("start", pl.Int64),
                ("end", pl.Int64),
                ("cm_mb", pl.Float64),
                ("cm", pl.Float64),
            ]
        ),
    )

    df_r_l = []
    nthreads = min(len(vcf_files), os.cpu_count() or 1)
    click.echo(click.style("Reading and sliding VCF file(s)", bold=True))
    with Parallel(n_jobs=nthreads, verbose=2) as parallel:
        df_r_l = parallel(delayed(read_regions)(i) for i in vcf_files)

    df_r = pl.concat(df_r_l)
    df_r = df_r.filter(pl.col("cm_mb") >= min_rate)

    lbl_fmt = ".2f"

    ps = [i / bins for i in range(1, bins)]

    qs_row = df_r.select(
        [pl.col("cm_mb").quantile(p).alias(f"q{i}") for i, p in enumerate(ps, start=1)]
    ).row(0)

    breaks = sorted({float(x) for x in qs_row if x is not None})

    hi = float(df_r.select(pl.col("cm_mb").max()).item())

    # Round intermediate edges normally
    edges = [round(x, 2) for x in breaks]

    # Force last edge to be >= true max (round UP, not nearest)
    hi_rounded_up = math.ceil(hi * 10) / 10  # → 21.2 in your case
    edges.append(hi_rounded_up)

    # Optional: ensure strictly increasing (important for binning)
    edges = sorted(set(edges))

    # Labels
    labels = [float(f"{x:{lbl_fmt}}") for x in edges]

    labels_str = ",".join(f"{x:{lbl_fmt}}" for x in edges)

    click.echo(click.style("Recombination bins:", bold=True) + f" {labels_str}")


@cli.command()
@click.option(
    "--train_data",
    type=str,
    required=False,
    help="Path to feature vectors from simulations for training the CNN.",
)
@click.option(
    "--predict_data",
    type=str,
    required=False,
    help="Path to feature vectors from empirical data for prediction.",
)
@click.option(
    "--center",
    type=str,
    required=False,
    default="0,1.2e6",
    help="Simulated start and end to sliding windows over the regions. Interally using the centers.",
)
@click.option(
    "--windows",
    type=int,
    required=False,
    default=int(1e5),
    help="Window size to sliding windows over the regions",
)
@click.option(
    "--step",
    type=int,
    required=False,
    default=int(1e5),
    help="Step size to sliding windows over the regions",
)
@click.option(
    "--output_folder",
    type=str,
    required=True,
    help="Directory to store the trained model, logs, and predictions.",
)
@click.option(
    "--model",
    type=str,
    default=None,
    help="Path to a pre-trained CNN model. If provided, the CNN will only perform prediction.",
)
def cnn(train_data, predict_data, output_folder, model):
    """
    Run the Flex-sweep CNN for training or prediction.

    Depending on the inputs the software train, predict or train/predict.

    \b
    Train example:
        flexsweep cnn --train_data data/train.parquet --output_folder ./sims/

    \b
    Predict example:
        flexsweep cnn --model ./sims/model.keras --predict_data data/test.parquet --output_folder results/
    \b
    Train/predict example:
        flexsweep cnn --train_data data/train.parquet --predict_data data/train.parquet --output_folder ./sims/

    """
    import flexsweep as fs

    os.makedirs(output_folder, exist_ok=True)

    center_list = parse_float_list(center)

    if model is None:
        if not train_data:
            raise click.UsageError(
                "--train_data is required when --model is not provided."
            )
        fs_cnn = fs.CNN(
            train_data=train_data,
            output_folder=output_folder,
            center=center_list,
            windows=windows,
            step=step,
        )
        fs_cnn.train()
        if predict_data is not None:
            fs_cnn.predict_data = predict_data
            fs_cnn.predict()

    else:
        if predict_data is None:
            raise click.UsageError(
                "--predict_data is required when --model is provided."
            )
        fs_cnn = fs.CNN(
            predict_data=predict_data,
            output_folder=output_folder,
            model=model,
            center=center_list,
            windows=windows,
            step=step,
        )
        fs_cnn.predict()


@cli.command()
@click.option(
    "--source_data",
    type=str,
    required=False,
    help="Path to feature vectors from simulations for training the CNN.",
)
@click.option(
    "--target_data",
    type=str,
    required=True,
    help="Path to feature vectors from empirical data for prediction.",
)
@click.option(
    "--center",
    type=str,
    required=False,
    default="0,1.2e6",
    help="Simulated start and end to sliding windows over the regions. Interally using the centers.",
)
@click.option(
    "--windows",
    type=int,
    required=False,
    default=int(1e5),
    help="Window size to sliding windows over the regions",
)
@click.option(
    "--step",
    type=int,
    required=False,
    default=int(1e5),
    help="Step size to sliding windows over the regions",
)
@click.option(
    "--target_ratio",
    type=float,
    required=False,
    default=2.0,
    help="Target ratio",
)
@click.option(
    "--max_lambda",
    type=float,
    required=False,
    default=1.0,
    help="Maximum labmda",
)
@click.option(
    "--ramp_epochs",
    type=float,
    required=False,
    default=30,
    help="Ramping epochs",
)
@click.option(
    "--batch_size",
    type=float,
    required=False,
    default=64,
    help="Batch size to train",
)
@click.option(
    "--output_folder",
    type=str,
    required=True,
    help="Directory to store the trained model, logs, and predictions.",
)
@click.option(
    "--preprocess",
    is_flag=True,
    default=False,
    help="Preprocessing stats",
)
@click.option(
    "--model",
    type=str,
    default=None,
    help="Path to a pre-trained DANN model. If provided, the DANN will only perform prediction.",
)
def dann(
    source_data,
    target_data,
    center,
    windows,
    step,
    target_ratio,
    max_lambda,
    ramp_epochs,
    batch_size,
    output_folder,
    preprocess,
    model,
):
    """
    Run the Flex-sweep DANN for training or prediction.

    Depending on the inputs the software train, predict or train/predict.

    \b
    Train example:
        flexsweep dann --source_data data/train.parquet --target_data data/empirical.parquet --output_folder ./sims/

    \b
    Predict example:
        flexsweep dann --model ./sims/model.keras --predict_data data/test.parquet --output_folder results/
    """
    import flexsweep as fs

    os.makedirs(output_folder, exist_ok=True)

    center_list = parse_float_list(center)

    if model is None:
        if not source_data:
            raise click.UsageError(
                "--source_data is required when --model is not provided."
            )
        fs_dann = fs.CNN(
            source_data=source_data,
            target_data=target_data,
            output_folder=output_folder,
            center=center_list,
            windows=fs.np.asarray([windows]),
            step=step,
        )
        fs_dann.train_da(
            tgt_ratio=target_ratio,
            max_lambda=max_lambda,
            ramp_epochs=ramp_epochs,
            batch_size=batch_size,
            preprocess=preprocess,
        )
        fs_dann.predict_da(preprocess=preprocess)
    else:
        if target_data is None:
            raise click.UsageError(
                "--target_data is required when --model is provided."
            )
        fs_dann = fs.CNN(
            source_data=source_data,
            target_data=target_data,
            output_folder=output_folder,
            model=model,
            center=center_list,
            windows=fs.np.asarray([windows]),
            step=step,
        )

        fs_dann.predict_da(preprocess=preprocess)


@cli.command()
@click.option(
    "--prediction",
    type=str,
    required=True,
    help="File containing predicted sweep probabilites.",
)
@click.option(
    "--feature_coordinates",
    type=str,
    required=True,
    help="Bed file contaning genomic feature coordinates.",
)
def rank(prediction, feature_coordinates):
    """
    Rank genomic features by their maximum nearby sweep probability.

    Read the predictions file and a BED of genomic features, links each
    feature to the nearest prediction window on the same chromosome, takes the **maximum**
    sweep probability among matched windows for that feature, ranks all features by this
    value.

    Input formats
    -------------
    * **prediction**: delimited text (TSV/CSV) with at least the columns
      ``chr``, ``start``, ``end``, ``prob_sweep``. Additional columns (e.g. ``iter``)
      are ignored for ranking.
    * **feature_coordinates**: BED-like, tab-separated with no header, columns
      ``chr``, ``start``, ``end``, and optionally an identifier (e.g. gene name) and
      strand. Coordinates are interpreted as 0-based half-open.

    :param prediction: Path to the file containing predicted sweep probabilities.
    :type prediction: str
    :param feature_coordinates: Path to a BED file with genomic feature intervals.
    :type feature_coordinates: str

    :returns: None. Writes the ranked table to ``output_file`` and prints number of max ranks.
    :rtype: None

    :notes:
        The ranking and feature–window association are performed by
        :func:`flexsweep.rank_probabilities`. Ties are reported as the count of features
        sharing the global maximum probability.
    """
    import os

    import flexsweep as fs

    df_rank, max_rank = fs.rank_probabilities(prediction, feature_coordinates)

    if max_rank > 1:
        click.echo(
            click.style(f"{max_rank}", bold=True)
            + " genomic features has the same highest sweep probability"
        )

    dir_name = os.path.dirname(prediction)
    base_name = os.path.splitext(os.path.basename(prediction))[0]

    output_path = os.path.join(dir_name, f"{base_name}_rank_{max_rank}.txt")

    df_rank[:, :2].write_csv(output_path, separator="\t", include_header=False)


@cli.command()
@click.option(
    "--sweep_files",
    type=str,
    required=True,
    help="Comma-separated paths to sweep rank files (gene_id + per-population rank columns).",
)
@click.option(
    "--gene_set",
    type=str,
    required=True,
    help="TSV with gene_id and yes/no label columns (no header).",
)
@click.option(
    "--factors",
    type=str,
    required=True,
    help="TSV confounding factors file (gene_id + factor columns, no header).",
)
@click.option(
    "--annotation",
    type=str,
    required=True,
    help="BED gene coordinates file (0-based, no header).",
)
@click.option(
    "--populations",
    type=str,
    required=True,
    help="Comma-separated population codes matching sweep file column order.",
)
@click.option(
    "--groups",
    type=str,
    required=True,
    help="Comma-separated group label per population (same length as --populations).",
)
@click.option(
    "--thresholds",
    type=str,
    required=True,
    help="Comma-separated rank thresholds (e.g. 6000,5000,...,20).",
)
@click.option(
    "--pop_interest",
    type=str,
    default="All",
    help="Population, group name, or 'All' for FDR scope.",
)
@click.option(
    "--cluster_distance",
    type=int,
    default=500_000,
    help="Max bp between genes to count as neighbours.",
)
@click.option(
    "--n_runs",
    type=int,
    default=10,
    help="Bootstrap batches (total sets = n_runs × iterations_per_run).",
)
@click.option(
    "--tolerance",
    type=float,
    default=0.05,
    help="Allowed ± fraction deviation in confounding factor averages.",
)
@click.option(
    "--min_distance",
    type=int,
    default=1_250_000,
    help="Minimum bp distance from VIPs for control gene eligibility.",
)
@click.option(
    "--flip",
    is_flag=True,
    default=False,
    help="Flip test direction when |VIPs| > |controls| (increases power).",
)
@click.option(
    "--max_rep",
    type=int,
    default=25,
    help="Max average resamples per control gene across all bootstrap sets.",
)
@click.option(
    "--nthreads",
    type=int,
    default=1,
    help="Parallel workers (joblib).",
)
@click.option(
    "--n_shuffles",
    type=int,
    default=8,
    help="FDR shuffle replicates (must be a multiple of 8).",
)
@click.option(
    "--shuffling_segs",
    type=int,
    default=2,
    help="Genes per genomic shuffle segment.",
)
@click.option(
    "--distance_file",
    type=str,
    default=None,
    help="Pre-computed gene set distance",
)
@click.option(
    "--facet",
    is_flag=True,
    default=False,
    help="Facet populations in enrichment plot",
)
@click.option(
    "--plot_groups",
    is_flag=True,
    default=False,
    help="Facet populations in enrichment plot",
)
@click.option(
    "--output_folder",
    type=str,
    default=None,
    help="Folder to output files).",
)
def enrichment(
    sweep_files,
    gene_set,
    factors,
    annotation,
    populations,
    groups,
    thresholds,
    pop_interest,
    cluster_distance,
    n_runs,
    tolerance,
    min_distance,
    flip,
    max_rep,
    nthreads,
    n_shuffles,
    shuffling_segs,
    distance_file,
    facet,
    plot_groups,
    output_folder,
):
    """
    Gene-set sweep enrichment and FDR analysis.

    Tests whether a set of genes of interest (VIPs) shows a significant excess
    or deficit of positive selection signals relative to matched control gene
    sets. Runs bootstrap control set generation, sweep counting across rank
    thresholds, genome shuffling for a null distribution, and FDR estimation.

    Input files
    -----------
    * **sweep_files**: tab- or space-separated, optionally gzipped. First column
      ``gene_id``; remaining columns are per-population sweep ranks matching
      ``--populations`` order.
    * **gene_set**: two-column TSV (no header) ``gene_id``, ``yes``/``no``.
    * **factors**: confounding factors TSV (no header); first column ``gene_id``.
    * **annotation**: BED file (0-based, no header) ``chr``, ``start``, ``end``,
      ``gene_id``.

    \b
    Example usage:
        flexsweep enrichment \\
            --sweep_files yri_ranks.tsv \\
            --gene_set vip_genes.tsv \\
            --factors confounders.tsv \\
            --annotation genes.bed \\
            --populations YRI \\
            --groups AFR \\
            --thresholds 6000,5000,2000,1000,500,200,100,50,20 \\
            --nthreads 8
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    from flexsweep.enrichment import np, pl, run_enrichment

    NPG_COLORS = [
        "#E64B35",
        "#4DBBD5",
        "#00A087",
        "#3C5488",
        "#F39B7F",
        "#8491B4",
        "#91D1C2",
        "#DC0000",
        "#7E6148",
        "#B09C85",
    ]

    def enrichment_plot_python(
        df_curves,
        plot_title: str = "",
        ci: bool = False,
        population: list[str] | None = None,
        target_groups: list[str] | None = None,
        facet: bool = False,
    ):
        df = df_curves.with_columns(
            [
                pl.col("ratio").cast(pl.Float64, strict=False),
                pl.col("threshold").cast(pl.Float64, strict=False),
                pl.col("ci_lo_ratio").cast(pl.Float64, strict=False),
                pl.col("ci_hi_ratio").cast(pl.Float64, strict=False),
                pl.col("vip_count").cast(pl.Float64, strict=False),
            ]
        )

        filters = [
            pl.col("threshold").is_not_null(),
            pl.col("ratio") >= 0,
        ]

        if population is not None:
            filters.append(pl.col("scope").is_in(population))

        if target_groups is not None:
            filters.append(pl.col("scope").is_in(target_groups))

        df = df.filter(pl.all_horizontal(filters))

        if ci:
            df = df.with_columns(
                [
                    (pl.col("ci_lo_ratio") / pl.col("vip_count")).alias("CI_low_ratio"),
                    (pl.col("ci_hi_ratio") / pl.col("vip_count")).alias("CI_up_ratio"),
                ]
            )
        else:
            df = df.with_columns(
                [
                    pl.col("ci_lo_ratio").alias("CI_low_ratio"),
                    pl.col("ci_hi_ratio").alias("CI_up_ratio"),
                ]
            )

        df = df.sort("threshold")

        threshold_levels = sorted(df["threshold"].unique().to_list())
        threshold_labels = [
            str(int(t)) if t == int(t) else str(t) for t in threshold_levels
        ]
        label_to_x = {lbl: i for i, lbl in enumerate(threshold_labels)}

        df = df.with_columns(
            pl.col("threshold")
            .map_elements(
                lambda v: label_to_x[str(int(v)) if v == int(v) else str(v)],
                return_dtype=pl.Int32,
            )
            .alias("x_pos")
        )

        scopes = df["scope"].unique().sort().to_list()
        color_map = {
            scope: NPG_COLORS[i % len(NPG_COLORS)] for i, scope in enumerate(scopes)
        }

        if facet:
            pops = scopes
            ncols = min(3, len(pops))
            nrows = int(np.ceil(len(pops) / ncols))

            fig, axes = plt.subplots(
                nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False
            )
            ax_flat = axes.flatten()

            for idx, pop in enumerate(pops):
                ax = ax_flat[idx]
                pop_df = df.filter(pl.col("scope") == pop)

                _draw_lines(ax, pop_df, scopes, color_map, ci)
                _style_ax(ax, threshold_labels, subtitle=pop)

            for idx in range(len(pops), len(ax_flat)):
                ax_flat[idx].set_visible(False)

            handles = [
                plt.Line2D(
                    [0], [0], color=color_map[s], linewidth=2, marker="o", label=s
                )
                for s in scopes
            ]
            fig.legend(
                handles=handles, title="Population", loc="lower right", frameon=False
            )

        else:
            fig, ax = plt.subplots(figsize=(9, 5))

            _draw_lines(ax, df, scopes, color_map, ci)
            _style_ax(ax, threshold_labels)

            handles = [
                plt.Line2D(
                    [0], [0], color=color_map[s], linewidth=2, marker="o", label=s
                )
                for s in scopes
            ]
            ax.legend(handles=handles, title="Population", frameon=False)

        fig.suptitle(plot_title, fontsize=14, fontweight="bold", y=1.01)
        fig.text(
            0.5,
            0.98,
            "Rank Threshold vs. Enrichment Ratio",
            ha="center",
            fontsize=10,
            color="grey",
            transform=fig.transFigure,
        )

        fig.tight_layout()
        return fig

    def _draw_lines(ax, df, scopes, color_map, ci):
        """Plot one line per scope onto *ax*."""
        ax.set_yscale("log")
        ax.axhline(y=1, linestyle="--", color="grey", linewidth=0.8, zorder=0)

        for scope in scopes:
            sub = df.filter(pl.col("scope") == scope).sort("x_pos")
            if sub.is_empty():
                continue
            x = sub["x_pos"].to_list()
            y = sub["ratio"].to_list()
            color = color_map[scope]

            ax.plot(
                x, y, color=color, linewidth=1.5, marker="o", markersize=4, label=scope
            )

            if ci:
                y_lo = sub["CI_low_ratio"].to_list()
                y_hi = sub["CI_up_ratio"].to_list()
                ax.fill_between(x, y_lo, y_hi, color=color, alpha=0.15, linewidth=0)

    def _style_ax(ax, threshold_labels, subtitle=None):
        """Apply common axis styling."""
        ax.set_xticks(range(len(threshold_labels)))
        ax.set_xticklabels(threshold_labels, rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("Rank Threshold")
        ax.set_ylabel("Enrichment Ratio (Obs/Exp)")

        ax.yaxis.set_major_locator(
            ticker.LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 5.0), numticks=20)
        )
        # ScalarFormatter: shows "10" instead of "10^1"
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

        # # Optional: Minor ticks for visual density without labels
        ax.yaxis.set_minor_locator(
            ticker.LogLocator(base=10.0, subs="auto", numticks=20)
        )
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)

        ax.spines[["top", "right"]].set_visible(False)
        if subtitle:
            ax.set_title(subtitle, fontsize=10)

    df_case, df_control, df_curves, df_fdr = run_enrichment(
        sweep_files=sweep_files.split(","),
        gene_set=gene_set,
        factors_file=factors,
        annotation_file=annotation,
        populations=populations.split(","),
        groups=groups.split(","),
        thresholds=[int(x) for x in thresholds.split(",")],
        pop_interest=pop_interest,
        cluster_distance=cluster_distance,
        n_runs=n_runs,
        tolerance=tolerance,
        min_distance=min_distance,
        flip=flip,
        max_rep=max_rep,
        nthreads=nthreads,
        n_shuffles=n_shuffles,
        shuffling_segs=shuffling_segs,
        distance_file=distance_file,
    )

    out_p = {}
    for k, v in df_curves.group_by("dataset"):
        out_p[k[0]] = enrichment_plot_python(
            v,
            facet=facet,
            population=np.unique(groups.split(","))
            if plot_groups
            else np.unique(populations.split(",")),
        )

    if output_folder is not None:
        df_case.write_csv(f"{output_folder}/case_set.txt")
        df_control.write_csv(f"{output_folder}/control_set.txt")
        df_curves.write_csv(f"{output_folder}/enrichment_curves.txt")
        df_fdr.write_csv(f"{output_folder}/fdr.txt")

        for k, v in out_p.items():
            v.savefig(f"{output_folder}/enrichement_curves_{k}.svg")
    else:
        plt.show()


@cli.command()
@click.option(
    "--maf",
    type=str,
    required=True,
    help=" Input MAF file (sorted by contig/pos). Can be gzipped",
)
@click.option(
    "--vcf",
    type=str,
    required=True,
    help=" Input VCF file (sorted by contig/pos). Can be gzipped",
)
@click.option(
    "--outgroups",
    type=str,
    required=True,
    help="Comma-separated list of outgroups, e.g: panTro4,ponAbe2,gorGor6",
)
@click.option(
    "--method",
    type=str,
    required=True,
    help="Method to polarize VCF file [possible values: parsimony, jc, kimura, r6]",
)
@click.option(
    "--nrandom",
    type=str,
    required=False,
    default=10,
    help="Number of random starts",
)
def polarize(maf, vcf, outgroups, method, nrandom):
    """
    Polarize VCF using rust est-sfs refactor. MAF file anchored to the VCF specie will be used to parse outgroups information.
    Please cite: https://doi.org/10.1534/genetics.118.301120

    Polarization is a streaming merge-join, so both inputs must contain a SINGLE reference
    contig, sorted by position, and must name that contig identically. Prepare them first:

    \b
        flexsweep split-maf --maf primates.maf.gz --contig CM000663.2
        bcftools view -r CM000663.2 all.vcf.gz -Oz -o yri_CM000663.2.vcf.gz

    \b
    Example usage:
        flexsweep polarize --maf primates_CM000663.2.maf.gz --vcf yri_CM000663.2.vcf.gz --outgroups specie1,specie2,specie3 --method kimura

    """

    import flexsweep as fs
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        fs.polarize.polarize(maf, vcf, outgroups, method, int(nrandom))
    except Exception as e:
        click.secho(f"Error during polarization: {e}", fg="red")
        raise click.Abort()


@cli.command(name="split-maf")
@click.option(
    "--maf",
    type=str,
    required=True,
    help="Input MAF file to split. Can be gzipped.",
)
@click.option(
    "--contig",
    type=str,
    required=True,
    help="Reference contig to extract, e.g: CM000663.2. Must match the VCF's contig name.",
)
@click.option(
    "--outdir",
    type=str,
    required=False,
    default=None,
    help="Output directory. Defaults to the input MAF's directory.",
)
@click.option(
    "--sort/--no-sort",
    default=True,
    help="Sort the extracted contig by position. On by default: polarize requires it.",
)
@click.option(
    "--tmp-dir",
    type=str,
    required=False,
    default=None,
    help="Directory for the intermediate split and the sort's spill chunks.",
)
@click.option(
    "--chunk-bytes",
    type=int,
    required=False,
    default=268435456,
    help="Maximum bytes held in memory before the sort spills to disk.",
)
def split_maf(maf, contig, outdir, sort, tmp_dir, chunk_bytes):
    """
    Split a MAF by reference contig and sort it by position, ready for polarize.

    \b
    Polarization is a streaming merge-join over the MAF and VCF: its cursor only moves
    forward and never rewinds, so an unsorted or multi-contig MAF makes it silently skip
    variants. This command produces the single-contig, position-sorted MAF it needs.

    \b
    The VCF side is a standard bcftools job, so flexsweep does not duplicate it:
        bcftools view -r CM000663.2 all.vcf.gz -Oz -o yri_CM000663.2.vcf.gz
        tabix -p vcf yri_CM000663.2.vcf.gz

    \b
    Example usage:
        flexsweep split-maf --maf primates.maf.gz --contig CM000663.2
    """

    import flexsweep as fs
    import logging
    import tempfile

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not os.path.isfile(maf):
        click.secho(f"Error: MAF file not found: {maf}", fg="red")
        raise click.Abort()

    base_name = os.path.basename(maf).replace(".maf.gz", "").replace(".maf", "")
    out_dir = outdir if outdir is not None else os.path.dirname(maf)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Named for the contig alone, whether or not sorting ran, so --no-sort output stays
    # drop-in interchangeable and the name never leaks which internal steps happened.
    maf_out = os.path.join(out_dir, f"{base_name}_{contig}.maf.gz")

    scratch = tmp_dir if tmp_dir is not None else tempfile.gettempdir()
    os.makedirs(scratch, exist_ok=True)

    try:
        if sort:
            # The unsorted extract is a full contig of MAF and can run to many GB, so it
            # goes to scratch and is dropped once the sort has consumed it.
            fd, split_tmp = tempfile.mkstemp(
                prefix=f"{base_name}_{contig}_", suffix=".maf.gz", dir=scratch
            )
            os.close(fd)
            try:
                click.echo(f"Splitting MAF for contig: {contig}")
                fs.polarize.maf_split(maf, split_tmp, contig)

                click.echo("Sorting by position...")
                fs.polarize.maf_sort(
                    split_tmp, maf_out, None, int(chunk_bytes), scratch
                )
            finally:
                if os.path.exists(split_tmp):
                    os.unlink(split_tmp)
        else:
            click.echo(f"Splitting MAF for contig: {contig}")
            fs.polarize.maf_split(maf, maf_out, contig)
    except Exception as e:
        click.secho(f"Error during split-maf: {e}", fg="red")
        raise click.Abort()

    click.secho(f"Wrote {maf_out}", fg="green")


@cli.command()
@click.option(
    "--vcf_path",
    "vcf_path",
    type=str,
    required=True,
    help="Directory of *.vcf.gz files, one per chromosome/contig.",
)
@click.option(
    "--out_prefix",
    "out_prefix",
    type=str,
    required=True,
    help="Output prefix. Writes {out}.{stat}.txt for each selected stat.",
)
@click.option(
    "--stats",
    type=str,
    required=True,
    help="Comma-separated stat keys, e.g. ihs,nsl,h12,lassip,raisd. "
    "Run 'python -c \"from flexsweep.scan import available_stats; "
    "print(available_stats())\"' to list all options.",
)
@click.option(
    "--w_size",
    type=int,
    default=201,
    show_default=True,
    help="SNP-count window size for sliding-window stats.",
)
@click.option(
    "--step",
    type=int,
    default=10,
    show_default=True,
    help="SNP step for sliding-window stats.",
)
@click.option(
    "--min_maf",
    type=float,
    default=0.05,
    show_default=True,
    help="Minimum minor allele frequency for iHS and nSL.",
)
@click.option(
    "--recombination_map",
    type=str,
    default=None,
    help="TSV recombination map (chr, start, end, cm_mb, cm). "
    "Enables genetic-distance windows and joint DAF+recomb normalization.",
)
@click.option(
    "--n_daf_bins",
    type=int,
    default=100,
    show_default=True,
    help="Number of DAF bins for iHS/nSL normalization.",
)
@click.option(
    "--max_extend",
    type=float,
    default=1e5,
    show_default=True,
    help="saltiLASSI spatial decay cutoff in bp.",
)
@click.option(
    "--K_truncation",
    "K_truncation",
    type=int,
    default=10,
    show_default=True,
    help="K truncation for LASSI/saltiLASSI.",
)
@click.option(
    "--sweep_mode",
    type=int,
    default=4,
    show_default=True,
    help="Sweep model for LASSI/saltiLASSI (1-5).",
)
@click.option(
    "--raisd_window",
    type=int,
    default=50,
    show_default=True,
    help="SNP window size for RAISD.",
)
@click.option(
    "--nthreads",
    type=int,
    default=1,
    show_default=True,
    help="Total worker threads for the global task pool. Window-batchable "
    "stats split each chromosome into nthreads window batches.",
)
@click.option(
    "--window_mode",
    default="auto",
    type=click.Choice(["auto", "snp", "bp"]),
    help="Window mode for sliding-window stats. 'auto' uses per-stat defaults "
    "(h12/garud/lassi/lassip/raisd=snp; neutrality/omega/beta=bp).",
)
@click.option(
    "--w_size_bp",
    default=1_000_000,
    type=int,
    help="Physical window size in bp for bp-mode stats (default 1Mb).",
)
@click.option(
    "--step_bp",
    default=10_000,
    type=int,
    help="Physical step size in bp for bp-mode stats (default 10kb).",
)
@click.option(
    "--window_size",
    default=50_000,
    type=int,
    show_default=True,
    help="Focal window size in bp for T3 per-SNP stats (dind, hapdaf_o, "
    "hapdaf_s, s_ratio, high_freq, low_freq).",
)
def scan(
    vcf_path,
    out_prefix,
    stats,
    w_size,
    step,
    min_maf,
    recombination_map,
    n_daf_bins,
    max_extend,
    K_truncation,
    sweep_mode,
    raisd_window,
    nthreads,
    window_mode,
    w_size_bp,
    step_bp,
    window_size,
):
    """Standalone outlier scan from a directory of per-chromosome VCF files.

    Computes selected statistics at their natural resolution (per-SNP or sliding
    window) and ranks each genome-wide. No neutral simulations required.
    Writes one tab-separated file per stat: {out}.{stat}.txt.

    Uses a global task pool: all chromosomes are loaded first, then all tasks
    (every stat × chromosome combination) are dispatched in one Parallel call
    to fully exploit --nthreads regardless of chromosome count.

    Global filters apply uniformly to all selected stats of the same class:
    --min_maf applies to both ihs and nsl; --window_size applies to dind,
    hapdaf_o, hapdaf_s, s_ratio, high_freq, and low_freq.

    \b
    Per-SNP stats:    ihs  nsl  isafe  dind  s_ratio  hapdaf_o  hapdaf_s  haf  hscan
    Window stats:     h12  garud  neutrality  omega  lassi  lassip  raisd  beta  ncd

    \b
    Examples:
        flexsweep scan --vcf_path vcf_folder/ --out YRI --stats ihs,nsl
        flexsweep scan --vcf_path vcf_folder/ --out YRI --stats ihs,nsl,h12,lassip --nthreads 8
        flexsweep scan --vcf_path vcf_folder/ --out YRI \\
            --stats dind,hapdaf_o,hapdaf_s --window_size 100000
    """
    from flexsweep.scan import scan as _scan

    stat_list = [s.strip() for s in stats.split(",")]
    _scan(
        vcf_path=vcf_path,
        out_prefix=out_prefix,
        stats=stat_list,
        w_size=w_size,
        step=step,
        w_size_bp=w_size_bp,
        step_bp=step_bp,
        min_maf=min_maf,
        recombination_map=recombination_map,
        n_daf_bins=n_daf_bins,
        nthreads=nthreads,
        window_mode=window_mode,
        config={
            "lassi": {"K_truncation": K_truncation, "sweep_mode": sweep_mode},
            "lassip": {
                "K_truncation": K_truncation,
                "sweep_mode": sweep_mode,
                "max_extend": max_extend,
            },
            "raisd": {"window_size": raisd_window},
            "dind": {"window_size": window_size},
            "high_freq": {"window_size": window_size},
            "low_freq": {"window_size": window_size},
            "s_ratio": {"window_size": window_size},
            "hapdaf_o": {"window_size": window_size},
            "hapdaf_s": {"window_size": window_size},
        },
    )


@cli.command("scan-plot")
@click.option(
    "--stat",
    "stat_files",
    type=str,
    multiple=True,
    required=True,
    help="Path(s) to scan output file(s). Repeat for multiple stats.",
)
@click.option(
    "--col",
    "stat_cols",
    type=str,
    multiple=True,
    required=True,
    help="Column name(s) to plot, matching order of --stat.",
)
@click.option(
    "--pvalue",
    is_flag=True,
    default=False,
    help="Plot the {col}_rank column instead of raw values.",
)
@click.option(
    "--top_pct",
    type=float,
    default=0.01,
    show_default=True,
    help="Fraction of top-ranked points to highlight (default 0.01 = 1%).",
)
@click.option(
    "--multi",
    is_flag=True,
    default=False,
    help="Stacked multi-stat plot with shared x-axis.",
)
@click.option(
    "--out",
    "out_file",
    type=str,
    default=None,
    help="Output image path (png/pdf/svg). If omitted, shows interactively.",
)
def scan_plot(stat_files, stat_cols, pvalue, top_pct, multi, out_file):
    """Visualize outlier scan results.

    \b
    Single stat:
        flexsweep scan-plot --stat YRI.chr22.lassip.txt --col Lambda
    Multi-stat stacked tracks:
        flexsweep scan-plot --stat YRI.chr22.ihs.txt --col ihs \\
            --stat YRI.chr22.h12.txt --col h12
    """
    from flexsweep.utils import plot_scan

    stat_files = list(stat_files)
    stat_cols = list(stat_cols)
    if len(stat_files) != len(stat_cols):
        raise click.UsageError("Number of --stat and --col arguments must match.")
    else:
        plot_scan(stat_files, stat_cols, pvalue=pvalue, top_pct=top_pct, out=out_file)


if __name__ == "__main__":
    cli()
