//! Contains methods and structs needed to compute various statistics about [ILP allocation](super).

use crate::allocation::ilp::SegmentId;
use crate::RequestId;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

/// Contains a data structure with information about the segments in any case (via [SegmentId]),
/// and, if the dp-type is rdp, information about the number of alphas and by how much this number
/// was reduced in local reduction.
#[derive(Clone, Debug)]
pub(crate) enum SegmentInfos {
    /// If there were no contested segments, regardless of dp-type
    NoSegments,
    /// >= 1 contested segments, and the dp-type was rdp, with more than one alpha value
    RdpSegInfos(BTreeMap<SegmentId, SegmentRdpInfo>),
    /// >= 1 contested segments, and the dp-type was eps-dp, adp, or rdp with just one alpha value
    /// since if there is only one alpha value, we build the ilp for eps-dp, and not the version
    /// for rdp.
    NonRdpSegInfos(BTreeSet<SegmentId>),
}

/// Contains information about which requests that were allocated by the ilp failed to be allocated
/// by the problem formulation, which requests were allocated by greedy afterwards.
#[derive(Clone, Debug, Serialize)]
pub struct FailedAndRetried {
    /// Contains all requests where allocation failed even though they were allocated by the ilp
    pub failed_allocations: BTreeSet<RequestId>,
    /// Contains all requests that were allocated greedily (if one or more requests allocated
    /// by the ilp could not be allocated by the problem formulation)
    ///
    /// Note that this may include requests that are not part of
    /// failed_allocations
    pub greedily_allocated: BTreeSet<RequestId>,
}

/// Contains the (arithmetic) mean/average and std deviation of some quantity
#[derive(Clone, Copy, Debug, Serialize)]
pub struct MeanAndStddev {
    pub(crate) mean: f64,
    pub(crate) stddev: f64,
}

impl MeanAndStddev {
    /// Initializes a new [Self] by calculating mean and stddev from the passed values.
    /// Assumes the passed values are samples from a larger population, and not the complete
    /// population.
    pub fn new(vals: &[f64]) -> Self {
        let n_vals = vals.len();
        let mean: f64 = vals.iter().sum::<f64>() / (n_vals as f64);
        let stddev: f64 = (vals
            .iter()
            .map(|val| (*val - mean).abs().powi(2))
            .sum::<f64>()
            / ((n_vals as f64) - 1.))
            .sqrt();
        MeanAndStddev { mean, stddev }
    }
}

/// Contains statistics about the ilp model, the size of the problem in terms of privacy resources,
/// and  optimizations done during model building (e.g.,
/// [alpha reduction](crate::dprivacy::rdpopt::RdpOptimization::calc_needed_alphas))
#[derive(Clone, Debug, Serialize)]
pub struct IlpStats {
    /// How many variables
    pub num_vars: i32,
    /// How many integer variables
    pub num_int_vars: i32,
    /// How many binary variables
    pub num_bin_vars: i32,
    /// How many constraints
    pub num_constr: i32,
    /// How many nonzero coefficients in constraint matrix
    pub num_nz_coeffs: f64,
    /// How many blocks were taken into account
    pub num_blocks: usize,
    /// How many contested segments there were in the problem formulation
    pub num_contested_segments_initially: usize,
    /// How many contested segments there were in the ilp
    pub ilp_num_contested_segments: usize,
    /// The mean and stddev of the contested segments per block
    pub contested_segments_per_block: MeanAndStddev,
    /// Contains all stats only present if rdp is enabled
    pub rdp_stats: Option<IlpRdpStats>,
    /// Contains which ilp allocation decisions could not be allocated by the problem formulation,
    /// and which requests were allocated greedily afterwards.
    pub failed_and_retried: Option<FailedAndRetried>,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct IlpRdpStats {
    /// how many alphas there are in total without local reduction (if the costs and budgets are
    /// in rdp)
    pub n_alphas_no_local_reduction: usize,
    /// how many alphas there are in total with local reduction (if the costs and budgets are
    /// in rdp)
    pub n_alphas_after_local_reduction: usize,
    /// how many alphas were reduced away by the budget reduction (if he costs and budgets are in
    /// rdp and the reduction was activated)
    pub n_alphas_eliminated_budget_reduction: Option<MeanAndStddev>,
    /// how many alphas were reduced away by the ratio reduction (if he costs and budgets are in
    /// rdp and the reduction was activated)
    pub n_alphas_eliminated_ratio_reduction: Option<MeanAndStddev>,
    /// how many alphas were reduced away by the combinatorial reduction (if he costs and budgets
    /// are in rdp and the reduction was activated)
    pub n_alphas_eliminated_combinatorial_reduction: Option<MeanAndStddev>,
}

/// Contains some information for a segment with rdp cost and budget
#[derive(Clone, Debug)]
pub(crate) struct SegmentRdpInfo {
    /// The infos about the individual constraint in the segment
    pub(crate) constraint_infos: Vec<ConstraintRdpInfo>,
}

/// Contains some information for a constraint of a segment with rdp cost and budget
#[derive(Clone, Copy, Debug)]
pub struct ConstraintRdpInfo {
    /// How many alphas this segment had before local alpha reduction
    pub(crate) alphas_no_red: usize,
    /// How many alphas this segment had after local alpha reduction
    pub(crate) alphas_after_red: usize,
    /// How many alphas were reduced by budged reduction (if enabled)
    pub(crate) alpha_removed_budget_red: Option<usize>,
    /// How many alphas were reduced by ratio reduction (if enabled)
    pub(crate) alpha_removed_ratio_red: Option<usize>,
    /// How many alphas were reduced by combinatorial reduction (if enabled)
    pub(crate) alpha_removed_combinatorial_red: Option<usize>,
}
