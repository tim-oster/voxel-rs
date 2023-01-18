use std::ops::Deref;

pub struct Resource<T, E> {
    constructor: Box<dyn Fn() -> Result<T, E>>,
    content: T,
}

// TODO any way to not have the error type as part of the resource type?
impl<T, E> Resource<T, E> {
    pub fn new<F>(constructor: F) -> Result<Self, E>
        where F: Fn() -> Result<T, E> + 'static {
        match constructor() {
            Ok(content) => Ok(Resource {
                constructor: Box::new(constructor),
                content,
            }),
            Err(err) => Err(err),
        }
    }

    pub fn reload(&mut self) -> Result<(), E> {
        let result = self.constructor.as_ref()();
        if let Ok(content) = result {
            self.content = content;
            return Ok(());
        }
        return Err(result.err().unwrap());
    }
}

impl<T, E> Deref for Resource<T, E> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.content
    }
}

impl<T, E> Bind for Resource<T, E> where T: Bind {
    fn bind(&self) {
        self.content.bind();
    }

    fn unbind(&self) {
        self.content.unbind();
    }
}

pub trait Bind {
    fn bind(&self);
    fn unbind(&self);
}
