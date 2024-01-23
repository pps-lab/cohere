use clap::Parser;
use dp_planner_lib::dprivacy::rdp_alphas_accounting::RdpAlphas::*;
use dp_planner_lib::dprivacy::AccountingType::{EpsDeltaDp, EpsDp, Rdp};
use log::info;
use std::ffi::OsStr;
use std::path::PathBuf;

/// The purpose of this binary is to generate a file containing a specified amount of blocks
/// with the specified budget type (initialized to zero budget). This can then be used to run a
/// simulation on top. Note that a block file including a history cannot be generated this way,
/// but can be generated by first running this binary, and then running a simulation with some
/// requests on top.
fn main() {
    let mut builder = env_logger::Builder::from_default_env();
    builder.target(env_logger::Target::Stdout);
    builder.init();

    let args: Args = Args::parse();

    // Check output path
    {
        // did not supply empty output_path
        let mut copy = args.output_path.clone();
        assert!(copy.pop(), "Empty output path was supplied");
        // all parent directories exist
        assert!(
            copy.exists(),
            "A directory on the supplied output path either does not exist or is inaccessible"
        );
        // check that file ends in .json (via
        // https://stackoverflow.com/questions/45291832/extracting-a-file-extension-from-a-given-path-in-rust-idiomatically)
        assert_eq!(
            args.output_path.extension().and_then(OsStr::to_str),
            Some("json"),
            "output file needs to have \".json\" extension (no capital letters)"
        );
    }

    let budget = match args.budget_type {
        BudgetTypeEnum::EpsDp => EpsDp { eps: 0.0 },
        BudgetTypeEnum::EpsDeltaDp => EpsDeltaDp {
            eps: 0.0,
            delta: 0.0,
        },
        BudgetTypeEnum::Rdp(rdp_info) => {
            let rdp_alphas = match rdp_info.num_alphas {
                1 => A1([0.; 1]),
                2 => A2([0.; 2]),
                3 => A3([0.; 3]),
                4 => A4([0.; 4]),
                5 => A5([0.; 5]),
                7 => A7([0.; 7]),
                10 => A10([0.; 10]),
                13 => A13([0.; 13]),
                14 => A14([0.; 14]),
                15 => A15([0.; 15]),
                _ => panic!("Given number of alphas for rdp not supported"),
            };
            Rdp {
                eps_values: rdp_alphas,
            }
        }
    };
    dp_planner_lib::util::generate_and_write_blocks(
        0,
        args.n_blocks,
        budget.clone(),
        &args.output_path,
    )
    .expect("Writing blocks failed");

    info!(
        "Wrote {} blocks with budget {:?} to location {:?}",
        args.n_blocks, budget, args.output_path
    )
}

/// Simple program to greet a person
#[derive(Parser, Debug, Clone)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Number of blocks to generate
    #[clap(short = 'N', long, default_value = "10")]
    n_blocks: usize,

    /// Where to store the blocks
    #[clap(
        short = 'O',
        long,
        parse(from_os_str),
        value_name = "FILE",
        default_value = "./blocks.json"
    )]
    output_path: PathBuf,

    /// Budget type (all blocks are initialized to zero budget)
    #[clap(subcommand)]
    budget_type: BudgetTypeEnum,
}

#[derive(clap::Subcommand, Debug, Clone)]
enum BudgetTypeEnum {
    EpsDp,
    EpsDeltaDp,
    Rdp(RdpInfo),
}

#[derive(clap::Args, Debug, Clone, Copy)]
struct RdpInfo {
    num_alphas: usize,
}
