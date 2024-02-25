use std::ops::Deref;

pub struct Resource<T, E> {
    constructor: Box<dyn Fn() -> Result<T, E>>,
    content: T,
}

pub trait Constructor<T, E>: Fn() -> Result<T, E> + 'static {}

impl<X, T, E> Constructor<T, E> for X
    where X: Fn() -> Result<T, E> + 'static {}

/// Resource holds an actual resource and its constructor, allowing it to be reloaded in place.
impl<T, E> Resource<T, E> {
    pub fn new<F: Constructor<T, E>>(constructor: F) -> Result<Self, E> {
        match constructor() {
            Ok(content) => Ok(Self {
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
        Err(result.err().unwrap())
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
