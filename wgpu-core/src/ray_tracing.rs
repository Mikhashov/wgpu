// Ray tracing
// Major missing optimizations (no api surface changes needed):
// - use custom tracker to track build state
// - no forced rebuilt (build mode deduction)
// - lazy instance buffer allocation
// - maybe share scratch and instance staging buffer allocation
// - partial instance buffer uploads (api surface already designed with this in mind)
// - Batch BLAS read-backs (if it shows up in performance).
// - ([non performance] extract function in build (rust function extraction with guards is a pain))

use alloc::{boxed::Box, sync::Arc, vec::Vec};
use core::sync::atomic::Ordering;

#[cfg(feature = "serde")]
use macro_rules_attribute::apply;
use thiserror::Error;
use wgt::{
    error::{ErrorType, WebGpuError},
    AccelerationStructureGeometryFlags, BufferAddress, IndexFormat, VertexFormat,
};

#[cfg(feature = "serde")]
use crate::command::serde_object_reference_struct;
use crate::{
    command::{ArcReferences, EncoderStateError, IdReferences, ReferenceType},
    device::{DeviceError, MissingFeatures},
    id::{BlasId, BufferId, TlasId},
    resource::{
        Blas, BlasCompactCallback, BlasPrepareCompactResult, DestroyedResourceError,
        InvalidResourceError, MissingBufferUsageError, ResourceErrorIdent, Tlas,
    },
};

#[derive(Clone, Debug, Error)]
pub enum CreateBlasError {
    #[error(transparent)]
    Device(#[from] DeviceError),
    #[error(transparent)]
    MissingFeatures(#[from] MissingFeatures),
    #[error(
        "Only one of 'index_count' and 'index_format' was provided (either provide both or none)"
    )]
    MissingIndexData,
    #[error("Provided format was not within allowed formats. Provided format: {0:?}. Allowed formats: {1:?}")]
    InvalidVertexFormat(VertexFormat, Vec<VertexFormat>),
    #[error("Limit `max_blas_geometry_count` is {0}, but the BLAS had {1} geometries")]
    TooManyGeometries(u32, u32),
    #[error(
        "Limit `max_blas_primitive_count` is {0}, but the BLAS had a maximum of {1} primitives"
    )]
    TooManyPrimitives(u32, u32),
    #[error("AABB geometry stride {0} is invalid (must be >= {1} and a multiple of 8)")]
    InvalidAabbStride(BufferAddress, BufferAddress),
}

impl WebGpuError for CreateBlasError {
    fn webgpu_error_type(&self) -> ErrorType {
        match self {
            Self::Device(e) => e.webgpu_error_type(),
            Self::MissingFeatures(e) => e.webgpu_error_type(),
            Self::MissingIndexData
            | Self::InvalidVertexFormat(..)
            | Self::TooManyGeometries(..)
            | Self::TooManyPrimitives(..)
            | Self::InvalidAabbStride(..) => ErrorType::Validation,
        }
    }
}

#[derive(Clone, Debug, Error)]
pub enum CreateTlasError {
    #[error(transparent)]
    Device(#[from] DeviceError),
    #[error(transparent)]
    MissingFeatures(#[from] MissingFeatures),
    #[error("Flag {0:?} is not allowed on a TLAS")]
    DisallowedFlag(wgt::AccelerationStructureFlags),
    #[error("Limit `max_tlas_instance_count` is {0}, but the TLAS had a maximum of {1} instances")]
    TooManyInstances(u32, u32),
}

impl WebGpuError for CreateTlasError {
    fn webgpu_error_type(&self) -> ErrorType {
        match self {
            Self::Device(e) => e.webgpu_error_type(),
            Self::MissingFeatures(e) => e.webgpu_error_type(),
            Self::DisallowedFlag(..) | Self::TooManyInstances(..) => ErrorType::Validation,
        }
    }
}

/// Error encountered while attempting to do a copy on a command encoder.
#[derive(Clone, Debug, Error)]
pub enum BuildAccelerationStructureError {
    #[error(transparent)]
    EncoderState(#[from] EncoderStateError),

    #[error(transparent)]
    Device(#[from] DeviceError),

    #[error(transparent)]
    InvalidResource(#[from] InvalidResourceError),

    #[error(transparent)]
    DestroyedResource(#[from] DestroyedResourceError),

    #[error(transparent)]
    MissingBufferUsage(#[from] MissingBufferUsageError),

    #[error(transparent)]
    MissingFeatures(#[from] MissingFeatures),

    #[error(
        "Data range of {region_size} B starting at offset {offset} would overrun the size {buffer_size} of buffer {buffer_ident:?}"
    )]
    InsufficientBufferSize {
        buffer_ident: ResourceErrorIdent,
        offset: BufferAddress,
        region_size: BufferAddress,
        buffer_size: BufferAddress,
    },

    #[error(
        "Offset {offset}, computed as {count} times {stride} B, exceeds the maximum addressable offset 2^32 - 1 within buffer {buffer_ident:?}"
    )]
    OffsetLimitedTo4GB {
        buffer_ident: ResourceErrorIdent,
        offset: BufferAddress,
        count: BufferAddress,
        stride: BufferAddress,
    },

    #[error("Buffer {0:?} associated offset doesn't align with the index type")]
    UnalignedIndexBufferOffset(ResourceErrorIdent),

    #[error("Buffer {0:?} associated offset is unaligned")]
    UnalignedTransformBufferOffset(ResourceErrorIdent),

    #[error("Buffer {0:?} associated index count not divisible by 3 (count: {1}")]
    InvalidIndexCount(ResourceErrorIdent, u32),

    #[error("Buffer {0:?} associated data contains None")]
    MissingAssociatedData(ResourceErrorIdent),

    #[error(
        "Blas {0:?} build sizes to may be greater than the descriptor at build time specified"
    )]
    IncompatibleBlasBuildSizes(ResourceErrorIdent),

    #[error("Blas {0:?} flags are different, creation flags: {1:?}, provided: {2:?}")]
    IncompatibleBlasFlags(
        ResourceErrorIdent,
        AccelerationStructureGeometryFlags,
        AccelerationStructureGeometryFlags,
    ),

    #[error("Blas {0:?} build vertex count is greater than creation count (needs to be less than or equal to), creation: {1:?}, build: {2:?}")]
    IncompatibleBlasVertexCount(ResourceErrorIdent, u32, u32),

    #[error("Blas {0:?} vertex formats are different, creation format: {1:?}, provided: {2:?}")]
    DifferentBlasVertexFormats(ResourceErrorIdent, VertexFormat, VertexFormat),

    #[error("Blas {0:?} stride was required to be at least {1} but stride given was {2}")]
    VertexStrideTooSmall(ResourceErrorIdent, u64, u64),

    #[error("Blas {0:?} stride was required to be a multiple of {1} but stride given was {2}")]
    VertexStrideUnaligned(ResourceErrorIdent, u64, u64),

    #[error("Blas {0:?} index count was provided at creation or building, but not the other")]
    BlasIndexCountProvidedMismatch(ResourceErrorIdent),

    #[error("Blas {0:?} build index count is greater than creation count (needs to be less than or equal to), creation: {1:?}, build: {2:?}")]
    IncompatibleBlasIndexCount(ResourceErrorIdent, u32, u32),

    #[error("Blas {0:?} index formats are different, creation format: {1:?}, provided: {2:?}")]
    DifferentBlasIndexFormats(ResourceErrorIdent, Option<IndexFormat>, Option<IndexFormat>),

    #[error("Blas {0:?} is compacted and so cannot be built")]
    CompactedBlas(ResourceErrorIdent),

    #[error("Blas {0:?} build sizes require index buffer but none was provided")]
    MissingIndexBuffer(ResourceErrorIdent),

    #[error(
        "Tlas {0:?} an associated instances contains an invalid custom index (more than 24bits)"
    )]
    TlasInvalidCustomIndex(ResourceErrorIdent),

    #[error(
        "Tlas {0:?} has {1} active instances but only {2} are allowed as specified by the descriptor at creation"
    )]
    TlasInstanceCountExceeded(ResourceErrorIdent, u32, u32),

    #[error("Blas {0:?} has flag USE_TRANSFORM but the transform buffer is missing")]
    TransformMissing(ResourceErrorIdent),

    #[error("Blas {0:?} is missing the flag USE_TRANSFORM but the transform buffer is set")]
    UseTransformMissing(ResourceErrorIdent),
    #[error(
        "Tlas {0:?} dependent {1:?} is missing AccelerationStructureFlags::ALLOW_RAY_HIT_VERTEX_RETURN"
    )]
    TlasDependentMissingVertexReturn(ResourceErrorIdent, ResourceErrorIdent),

    #[error("Blas {0:?} geometry kind at creation does not match build (triangles vs AABBs)")]
    BlasGeometryKindMismatch(ResourceErrorIdent),

    #[error(
        "Blas {0:?} build AABB primitive count is greater than creation count (creation: {1}, build: {2})"
    )]
    IncompatibleBlasAabbPrimitiveCount(ResourceErrorIdent, u32, u32),

    #[error("Blas {0:?} AABB primitive offset must be a multiple of 8")]
    UnalignedAabbPrimitiveOffset(ResourceErrorIdent),

    #[error("Blas {0:?} AABB stride is invalid (must be >= {1} and a multiple of 8)")]
    InvalidAabbStride(ResourceErrorIdent, BufferAddress),
}

impl WebGpuError for BuildAccelerationStructureError {
    fn webgpu_error_type(&self) -> ErrorType {
        match self {
            Self::EncoderState(e) => e.webgpu_error_type(),
            Self::Device(e) => e.webgpu_error_type(),
            Self::InvalidResource(e) => e.webgpu_error_type(),
            Self::DestroyedResource(e) => e.webgpu_error_type(),
            Self::MissingBufferUsage(e) => e.webgpu_error_type(),
            Self::MissingFeatures(e) => e.webgpu_error_type(),
            Self::InsufficientBufferSize { .. }
            | Self::OffsetLimitedTo4GB { .. }
            | Self::UnalignedIndexBufferOffset(..)
            | Self::UnalignedTransformBufferOffset(..)
            | Self::InvalidIndexCount(..)
            | Self::MissingAssociatedData(..)
            | Self::IncompatibleBlasBuildSizes(..)
            | Self::IncompatibleBlasFlags(..)
            | Self::IncompatibleBlasVertexCount(..)
            | Self::DifferentBlasVertexFormats(..)
            | Self::VertexStrideTooSmall(..)
            | Self::VertexStrideUnaligned(..)
            | Self::BlasIndexCountProvidedMismatch(..)
            | Self::IncompatibleBlasIndexCount(..)
            | Self::DifferentBlasIndexFormats(..)
            | Self::CompactedBlas(..)
            | Self::MissingIndexBuffer(..)
            | Self::TlasInvalidCustomIndex(..)
            | Self::TlasInstanceCountExceeded(..)
            | Self::TransformMissing(..)
            | Self::UseTransformMissing(..)
            | Self::TlasDependentMissingVertexReturn(..)
            | Self::BlasGeometryKindMismatch(..)
            | Self::IncompatibleBlasAabbPrimitiveCount(..)
            | Self::UnalignedAabbPrimitiveOffset(..)
            | Self::InvalidAabbStride(..) => ErrorType::Validation,
        }
    }
}

#[derive(Clone, Debug, Error)]
pub enum ValidateAsActionsError {
    #[error(transparent)]
    DestroyedResource(#[from] DestroyedResourceError),

    #[error("Tlas {0:?} is used before it is built")]
    UsedUnbuiltTlas(ResourceErrorIdent),

    #[error("Blas {0:?} is used before it is built (in Tlas {1:?})")]
    UsedUnbuiltBlas(ResourceErrorIdent, ResourceErrorIdent),

    #[error("Blas {0:?} is newer than the containing Tlas {1:?}")]
    BlasNewerThenTlas(ResourceErrorIdent, ResourceErrorIdent),

    #[error("Tlas {0:?} was built with only its changed instances, and another build of it was submitted since that was recorded")]
    TlasBuiltOutOfOrder(ResourceErrorIdent),
}

impl WebGpuError for ValidateAsActionsError {
    fn webgpu_error_type(&self) -> ErrorType {
        match self {
            Self::DestroyedResource(e) => e.webgpu_error_type(),
            Self::UsedUnbuiltTlas(..)
            | Self::UsedUnbuiltBlas(..)
            | Self::BlasNewerThenTlas(..)
            | Self::TlasBuiltOutOfOrder(..) => ErrorType::Validation,
        }
    }
}

#[derive(Debug)]
pub struct BlasTriangleGeometry<'a> {
    pub size: &'a wgt::BlasTriangleGeometrySizeDescriptor,
    pub vertex_buffer: BufferId,
    pub index_buffer: Option<BufferId>,
    pub transform_buffer: Option<BufferId>,
    pub first_vertex: u32,
    pub vertex_stride: BufferAddress,
    pub first_index: Option<u32>,
    pub transform_buffer_offset: Option<BufferAddress>,
}

#[derive(Debug)]
pub struct BlasAabbGeometry<'a> {
    pub size: &'a wgt::BlasAABBGeometrySizeDescriptor,
    pub stride: BufferAddress,
    pub aabb_buffer: BufferId,
    pub primitive_offset: u32,
}

pub enum BlasGeometries<'a> {
    TriangleGeometries(Box<dyn Iterator<Item = BlasTriangleGeometry<'a>> + 'a>),
    AabbGeometries(Box<dyn Iterator<Item = BlasAabbGeometry<'a>> + 'a>),
}

pub struct BlasBuildEntry<'a> {
    pub blas_id: BlasId,
    pub geometries: BlasGeometries<'a>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TlasBuildEntry {
    pub tlas_id: TlasId,
    pub instance_buffer_id: BufferId,
    pub instance_count: u32,
}

#[derive(Debug)]
pub struct TlasInstance<'a> {
    pub blas_id: BlasId,
    pub transform: &'a [f32; 12],
    pub custom_data: u32,
    pub mask: u8,
}

pub struct TlasPackage<'a> {
    pub tlas_id: TlasId,
    pub instances: Box<dyn Iterator<Item = Option<TlasInstance<'a>>> + 'a>,
    pub lowest_unmodified: u32,
    /// What changed since the caller's previous build of these instances, to
    /// convert and upload only that. `None` uploads every instance.
    pub changes: Option<TlasChanges>,
}

/// Which instances of a [`TlasPackage`] changed since the previous build
/// call made with the same `source`.
#[derive(Debug, Clone)]
pub struct TlasChanges {
    /// The caller's object that tracks the changes of one list of instances
    /// (never 0, unique among them).
    pub source: u64,
    /// Build calls made with `source` before this one.
    pub seq: u32,
    /// The instances (slots of [`TlasPackage::instances`]) that may differ
    /// from the ones `source` passed in its previous call; the number of
    /// slots never changes.
    pub dirty: core::ops::Range<u32>,
}

/// What one recorded build of a top level uploads and refers to.
#[derive(Debug)]
pub struct TlasBuildData {
    /// Hal instances to upload, and where in the instance buffer.
    pub(crate) bytes: Vec<u8>,
    pub(crate) offset: u64,
    /// Instances in the whole top level.
    pub(crate) instance_count: u32,
    pub(crate) dependencies: Arc<Vec<Arc<Blas>>>,
    /// Number of this build among the top level's recorded ones (from 1).
    pub(crate) serial: u64,
    /// A build that uploads a part of the instances: the serial of the build
    /// whose instances the buffer must hold when this one runs.
    pub(crate) base: Option<u64>,
}

/// The instances of a top level as of its last recorded build.
#[derive(Debug, Default)]
pub(crate) struct TlasRetained {
    /// [`TlasChanges::source`] and [`TlasChanges::seq`] of that build (0: none tracks them).
    pub(crate) source: u64,
    pub(crate) seq: u32,
    /// [`TlasBuildData::serial`] of that build.
    pub(crate) serial: u64,
    /// Each slot's bottom level and hal instance bytes.
    slots: Vec<Option<Arc<Blas>>>,
    bytes: Vec<u8>,
    /// Slots holding an instance, and their bottom levels in order.
    count: u32,
    dependencies: Arc<Vec<Arc<Blas>>>,
}

impl TlasRetained {
    pub(crate) fn slots(&self) -> usize {
        self.slots.len()
    }

    /// Takes `instances` as the slots from `first` on (`partial`), or as
    /// every slot, and returns the build that brings the instance buffer of
    /// `tlas` from the last recorded build's contents to these.
    pub(crate) fn apply(
        &mut self,
        tlas: &Tlas,
        first: usize,
        instances: &[Option<ArcTlasInstance>],
        partial: bool,
    ) -> TlasBuildData {
        let stride = tlas.device.alignments.raw_tlas_instance_size as usize;
        if !partial {
            self.slots.clear();
            self.slots.resize(instances.len(), None);
            self.bytes.clear();
            self.bytes.resize(instances.len() * stride, 0);
            self.count = 0;
        }
        // Whether the instances after these move in the buffer (it holds no
        // empty slots), and whether the list of bottom levels changes.
        let mut moved = !partial;
        let mut relinked = !partial;
        for (slot, instance) in (first..).zip(instances) {
            let was = &mut self.slots[slot];
            match instance {
                Some(instance) => {
                    let bytes = tlas.device.raw().tlas_instance_to_bytes(hal::TlasInstance {
                        transform: instance.transform,
                        custom_data: instance.custom_data,
                        mask: instance.mask,
                        blas_address: instance.blas.handle,
                        pipeline_intersection_data_offset: 0,
                    });
                    self.bytes[slot * stride..][..stride].copy_from_slice(&bytes);
                    match was {
                        Some(blas) if Arc::ptr_eq(blas, &instance.blas) => {}
                        Some(_) => relinked = true,
                        None => {
                            moved = true;
                            self.count += 1;
                        }
                    }
                    *was = Some(instance.blas.clone());
                }
                None => {
                    if was.take().is_some() {
                        moved = true;
                        self.count -= 1;
                    }
                }
            }
        }
        if moved || relinked {
            self.dependencies = Arc::new(self.slots.iter().flatten().cloned().collect());
        }

        let before = self.slots[..first].iter().flatten().count();
        let end = if moved {
            self.slots.len()
        } else {
            first + instances.len()
        };
        let mut bytes = Vec::with_capacity((end - first) * stride);
        for slot in first..end {
            if self.slots[slot].is_some() {
                bytes.extend_from_slice(&self.bytes[slot * stride..][..stride]);
            }
        }

        let base = partial.then_some(self.serial);
        self.serial = tlas.build_serial.fetch_add(1, Ordering::Relaxed) + 1;
        TlasBuildData {
            bytes,
            offset: (before * stride) as u64,
            instance_count: self.count,
            dependencies: self.dependencies.clone(),
            serial: self.serial,
            base,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct TlasBuild {
    pub tlas: Arc<Tlas>,
    pub dependencies: Arc<Vec<Arc<Blas>>>,
    /// The bottom levels of the instances this build uploaded, where those
    /// are not all of `dependencies` (a build of a part).
    pub uploaded: Option<Vec<Arc<Blas>>>,
    /// [`TlasBuildData::serial`] and [`TlasBuildData::base`]; serial 0 for a
    /// build wgpu did not record the instances of.
    pub serial: u64,
    pub base: Option<u64>,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct AsBuild {
    pub blas_s_built: Vec<Arc<Blas>>,
    pub tlas_s_built: Vec<TlasBuild>,
}

impl AsBuild {
    pub(crate) fn with_capacity(blas: usize, tlas: usize) -> Self {
        Self {
            blas_s_built: Vec::with_capacity(blas),
            tlas_s_built: Vec::with_capacity(tlas),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) enum AsAction {
    Build(AsBuild),
    UseTlas(Arc<Tlas>),
}

/// Like [`BlasTriangleGeometry`], but with owned data.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", apply(serde_object_reference_struct))]
pub struct OwnedBlasTriangleGeometry<R: ReferenceType> {
    pub size: wgt::BlasTriangleGeometrySizeDescriptor,
    pub vertex_buffer: R::Buffer,
    pub index_buffer: Option<R::Buffer>,
    pub transform_buffer: Option<R::Buffer>,
    pub first_vertex: u32,
    pub vertex_stride: BufferAddress,
    pub first_index: Option<u32>,
    pub transform_buffer_offset: Option<BufferAddress>,
}

pub type ArcBlasTriangleGeometry = OwnedBlasTriangleGeometry<ArcReferences>;
pub type TraceBlasTriangleGeometry = OwnedBlasTriangleGeometry<IdReferences>;

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", apply(serde_object_reference_struct))]
pub struct OwnedBlasAabbGeometry<R: ReferenceType> {
    pub size: wgt::BlasAABBGeometrySizeDescriptor,
    pub stride: BufferAddress,
    pub aabb_buffer: R::Buffer,
    pub primitive_offset: u32,
}

pub type ArcBlasAabbGeometry = OwnedBlasAabbGeometry<ArcReferences>;
pub type TraceBlasAabbGeometry = OwnedBlasAabbGeometry<IdReferences>;

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", apply(serde_object_reference_struct))]
pub enum OwnedBlasGeometries<R: ReferenceType> {
    TriangleGeometries(Vec<OwnedBlasTriangleGeometry<R>>),
    AabbGeometries(Vec<OwnedBlasAabbGeometry<R>>),
}

pub type ArcBlasGeometries = OwnedBlasGeometries<ArcReferences>;
pub type TraceBlasGeometries = OwnedBlasGeometries<IdReferences>;

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", apply(serde_object_reference_struct))]
pub struct OwnedBlasBuildEntry<R: ReferenceType> {
    pub blas: R::Blas,
    pub geometries: OwnedBlasGeometries<R>,
}

pub type ArcBlasBuildEntry = OwnedBlasBuildEntry<ArcReferences>;
pub type TraceBlasBuildEntry = OwnedBlasBuildEntry<IdReferences>;

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", apply(serde_object_reference_struct))]
pub struct OwnedTlasInstance<R: ReferenceType> {
    pub blas: R::Blas,
    pub transform: [f32; 12],
    pub custom_data: u32,
    pub mask: u8,
}

pub type ArcTlasInstance = OwnedTlasInstance<ArcReferences>;
pub type TraceTlasInstance = OwnedTlasInstance<IdReferences>;

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", apply(serde_object_reference_struct))]
pub struct OwnedTlasPackage<R: ReferenceType> {
    pub tlas: R::Tlas,
    /// The instances: every slot, or with `build` the ones it uploads.
    pub instances: Vec<Option<OwnedTlasInstance<R>>>,
    pub lowest_unmodified: u32,
    /// The build as recorded, `None` to upload `instances` as every slot.
    #[cfg_attr(feature = "serde", serde(skip))]
    pub build: Option<Arc<TlasBuildData>>,
}

pub type TraceTlasPackage = OwnedTlasPackage<IdReferences>;
pub type ArcTlasPackage = OwnedTlasPackage<ArcReferences>;

/// [`BlasTriangleGeometry`], without the resources.
#[derive(Debug, Clone)]
pub struct BlasTriangleGeometryInfo {
    pub size: wgt::BlasTriangleGeometrySizeDescriptor,
    pub first_vertex: u32,
    pub vertex_stride: BufferAddress,
    pub first_index: Option<u32>,
    pub transform_buffer_offset: Option<BufferAddress>,
}

#[derive(Clone, Debug, Error)]
pub enum BlasPrepareCompactError {
    #[error(transparent)]
    Device(#[from] DeviceError),
    #[error(transparent)]
    InvalidResource(#[from] InvalidResourceError),
    #[error("Compaction is already being prepared")]
    CompactionPreparingAlready,
    #[error("Cannot compact an already compacted BLAS")]
    DoubleCompaction,
    #[error("BLAS is not yet built")]
    NotBuilt,
    #[error("BLAS does not support compaction (is AccelerationStructureFlags::ALLOW_COMPACTION missing?)")]
    CompactionUnsupported,
}

impl WebGpuError for BlasPrepareCompactError {
    fn webgpu_error_type(&self) -> ErrorType {
        match self {
            Self::Device(e) => e.webgpu_error_type(),
            Self::InvalidResource(e) => e.webgpu_error_type(),
            Self::CompactionPreparingAlready
            | Self::DoubleCompaction
            | Self::NotBuilt
            | Self::CompactionUnsupported => ErrorType::Validation,
        }
    }
}

#[derive(Clone, Debug, Error)]
pub enum CompactBlasError {
    #[error(transparent)]
    Encoder(#[from] EncoderStateError),

    #[error(transparent)]
    Device(#[from] DeviceError),

    #[error(transparent)]
    InvalidResource(#[from] InvalidResourceError),

    #[error(transparent)]
    DestroyedResource(#[from] DestroyedResourceError),

    #[error(transparent)]
    MissingFeatures(#[from] MissingFeatures),

    #[error("BLAS was not prepared for compaction")]
    BlasNotReady,
}

impl WebGpuError for CompactBlasError {
    fn webgpu_error_type(&self) -> ErrorType {
        match self {
            Self::Encoder(e) => e.webgpu_error_type(),
            Self::Device(e) => e.webgpu_error_type(),
            Self::InvalidResource(e) => e.webgpu_error_type(),
            Self::DestroyedResource(e) => e.webgpu_error_type(),
            Self::MissingFeatures(e) => e.webgpu_error_type(),
            Self::BlasNotReady => ErrorType::Validation,
        }
    }
}

pub type BlasCompactReadyPendingClosure = (Option<BlasCompactCallback>, BlasPrepareCompactResult);
