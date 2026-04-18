use std::env;
use std::fs;
use std::process;

const VERSION: &str = env!("CARGO_PKG_VERSION");

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut input: Option<String> = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--input" | "-i" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("error: --input requires a path");
                    process::exit(2);
                }
                input = Some(args[i].clone());
            }
            "--version" | "-V" => {
                println!("t80_to_bullet {}", VERSION);
                return;
            }
            "--help" | "-h" => {
                print_help();
                return;
            }
            other => {
                eprintln!("error: unknown argument {}", other);
                print_help();
                process::exit(2);
            }
        }
        i += 1;
    }

    let Some(path) = input else {
        print_help();
        process::exit(2);
    };

    let meta = match fs::metadata(&path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error: stat {}: {}", path, e);
            process::exit(1);
        }
    };

    if !meta.is_file() {
        eprintln!("error: {} is not a regular file", path);
        process::exit(1);
    }

    println!("t80_to_bullet {} — smoke test", VERSION);
    println!("input:  {}", path);
    println!("bytes:  {}", meta.len());
    println!("note:   parser not yet implemented; this binary only validates the build flow.");
}

fn print_help() {
    eprintln!("usage: t80_to_bullet --input <path>");
    eprintln!();
    eprintln!("Convert Leela T80 v6 training data to Bullet's bulletformat.");
    eprintln!("This is a smoke-test stub; full parsing is not yet implemented.");
}
