use dp_planner_lib::run_program;

fn main() {
    let mut builder = env_logger::Builder::from_default_env();
    builder.target(env_logger::Target::Stdout);
    builder.init();
    run_program()
}
