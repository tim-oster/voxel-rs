use gl::types::GLsync;

use crate::gl_assert_no_error;

/// Fence is a low level synchronization object. Placing a fence will drop any previously placed
/// fence first and then immediately flush the new fence to the GPU command queue. After that,
/// calls to wait will block until the fence is signaled.
pub struct Fence {
    handle: Option<GLsync>,
}

impl Fence {
    pub fn new() -> Self {
        Self { handle: None }
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
                if result == gl::TIMEOUT_EXPIRED {
                    continue;
                }
                if result == gl::ALREADY_SIGNALED || result == gl::CONDITION_SATISFIED {
                    return;
                }
                gl_assert_no_error!();
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
