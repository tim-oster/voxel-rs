use std::env;
use std::ffi::OsString;
use std::fs;
use std::io::{Error, Write};
use std::iter::FromIterator;
use std::path::{Component, Path, PathBuf};

use rustc_hash::FxHashSet;
use walkdir::WalkDir;

/// build.rs builds an `asset_bundle.rs` file that contains the content of all files in the `/assets` directory
/// (expect for excluded dirs, such as `/assets/tests`), if feature `bundle-assets` is enabled.
fn main() {
    let env_cargo_manifest_dir = env::var_os("CARGO_MANIFEST_DIR").unwrap();
    let env_out_dir = env::var_os("OUT_DIR").unwrap();

    // ensure rebuilds are triggered on asset changes
    println!("cargo:rerun-if-changed=assets");

    let manifest_dir = Path::new(&env_cargo_manifest_dir);
    let bundle_path = Path::new(&env_out_dir).join("asset_bundle.rs");

    // start out with an empty asset list by default
    let mut asset_list = Vec::new();

    // if enabled, find all assets in directory
    if env::var_os("CARGO_FEATURE_BUNDLE_ASSETS") == Some(OsString::from("1")) {
        let skip_dirs = FxHashSet::from_iter(["tests".to_string()]);
        asset_list = find_assets("assets", skip_dirs);
    }

    generate_asset_bundle(manifest_dir, &bundle_path, asset_list).unwrap();
}

struct Asset {
    path: PathBuf,
    canonical_path: String,
}

fn find_assets(root_dir: &str, skip_dirs: FxHashSet<String>) -> Vec<Asset> {
    let mut found = Vec::new();

    let mut it = WalkDir::new(root_dir).into_iter();
    loop {
        let entry = it.next();
        if entry.is_none() {
            break;
        }
        let entry = entry.unwrap().unwrap();

        // ignore directories in general and skip files inside skipped dirs
        if entry.file_type().is_dir() {
            let name = entry.file_name().to_string_lossy().to_string();
            if skip_dirs.contains(&name) {
                it.skip_current_dir();
            }
            continue;
        }

        // use os-independent format to enable cross platform support
        let canonical_path = entry.path().components()
            .filter_map(|c| match c {
                Component::Normal(name) => Some(name.to_string_lossy().to_string()),
                _ => None,
            })
            .collect::<Vec<String>>()
            .join("/");

        found.push(Asset {
            path: entry.path().to_path_buf(),
            canonical_path,
        });
    }

    found
}

fn generate_asset_bundle(manifest_dir: &Path, dst_path: &Path, asset_list: Vec<Asset>) -> Result<(), Error> {
    let mut out_file = fs::File::create(dst_path)?;

    let mut var_names = Vec::new();
    for asset in &asset_list {
        let var_name = format!("INCL_{}", asset.canonical_path.to_uppercase().replace(['/', '.'], "_"));

        out_file.write_all(format!(
            "const {}: &[u8] = include_bytes!(r\"{}\");\n",
            var_name,
            manifest_dir.join(&asset.path).to_string_lossy(),
        ).as_bytes())?;

        var_names.push(var_name);
    }

    out_file.write_all(&[b'\n'])?;
    out_file.write_all(format!(
        "const BUNDLED_ASSETS: [(&str, &[u8]); {}] = [\n",
        asset_list.len(),
    ).as_bytes())?;

    for (i, asset) in asset_list.iter().enumerate() {
        out_file.write_all(format!(
            "\t(\"{}\", &{}),\n",
            asset.canonical_path,
            var_names[i],
        ).as_bytes())?;
    }

    out_file.write_all("];".as_bytes())?;
    out_file.flush()?;

    Ok(())
}
