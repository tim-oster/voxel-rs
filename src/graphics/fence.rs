use gl::types::GLsync;

pub struct Fence {
    handle: Option<GLsync>,
}

impl Fence {
    pub fn new() -> Fence {
        Fence { handle: None }
    }

    pub fn place(&mut self) {
        if self.handle.is_some() {
            unsafe { gl::DeleteSync(self.handle.unwrap()); }
        }
        let handle = unsafe { gl::FenceSync(gl::SYNC_GPU_COMMANDS_COMPLETE, 0) };
        self.handle = Some(handle);
    }

    pub fn wait(&self) {
        if self.handle.is_none() {
            return;
        }
        let lock = self.handle.unwrap();
        unsafe {
            loop {
                let result = gl::ClientWaitSync(lock, gl::SYNC_FLUSH_COMMANDS_BIT, 1);
                if result == gl::ALREADY_SIGNALED || result == gl::CONDITION_SATISFIED {
                    return;
                }
            }
        }
    }
}

impl Drop for Fence {
    fn drop(&mut self) {
        if self.handle.is_some() {
            unsafe { gl::DeleteSync(self.handle.unwrap()); }
        }
    }
}
