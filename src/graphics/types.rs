#[repr(align(16))]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct AlignedVec3<T>(pub cgmath::Vector3<T>);

#[repr(align(16))]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct AlignedVec4<T>(pub cgmath::Vector4<T>);

#[repr(align(8))]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct AlignedPoint2<T>(pub cgmath::Point2<T>);

#[repr(align(16))]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct AlignedPoint3<T>(pub cgmath::Point3<T>);

#[repr(align(4))]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct AlignedBool(pub bool);

