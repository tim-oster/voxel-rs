use std::io;

#[cfg(not(feature = "bundle-assets"))]
pub fn read(path: &str) -> io::Result<Vec<u8>> {
    use std::fs::File;
    use std::io::Read;

    let mut file = File::open(path)?;
    let size = file.metadata().map(|m| m.len() as usize).ok();
    let mut data = Vec::with_capacity(size.unwrap_or(0));
    file.read_to_end(&mut data)?;
    Ok(data)
}

#[cfg(feature = "bundle-assets")]
include!(concat!(env!("OUT_DIR"), "/asset_bundle.rs"));

#[cfg(feature = "bundle-assets")]
pub fn read(path: &str) -> io::Result<Vec<u8>> {
    use std::io::ErrorKind;

    for (p, data) in BUNDLED_ASSETS {
        if p == path {
            return Ok(data.to_vec());
        }
    }
    Err(io::Error::new(ErrorKind::NotFound, "file was not bundled during build"))
}
