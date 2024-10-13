use std::mem;

use parry::bounding_volume::{Aabb, BoundingVolume};
use rustc_hash::FxHashSet;

use crate::{
    data::Coarena,
    dynamics::RigidBodySet,
    geometry::{BroadPhase, BroadPhasePairEvent, ColliderHandle, ColliderPair, ColliderSet},
    math::Real,
};

/// A broad-phase using a sparse hierarchical grid
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Default)]
pub struct BroadPhaseSieveTree {
    tree: SieveTree<ColliderHandle>,
    meta: Coarena<ColliderMeta>,
}

impl BroadPhaseSieveTree {
    /// Create a new empty broad-phase
    pub fn new() -> Self {
        Self::default()
    }
}

impl BroadPhase for BroadPhaseSieveTree {
    fn update(
        &mut self,
        dt: Real,
        prediction_distance: Real,
        colliders: &mut ColliderSet,
        bodies: &RigidBodySet,
        modified_colliders: &[ColliderHandle],
        removed_colliders: &[ColliderHandle],
        events: &mut Vec<BroadPhasePairEvent>,
    ) {
        const ELEMENTS_PER_CELL: usize = 4;

        for &handle in removed_colliders {
            let meta = self.meta.remove(handle.0, ColliderMeta::default()).unwrap();
            let removed = self.tree.remove(meta.id, meta.bounds);
            debug_assert_eq!(removed, handle);
        }

        for &handle in modified_colliders {
            let co = colliders.get_mut_internal(handle).unwrap();
            if !co.is_enabled() || !co.changes.needs_broad_phase_update() {
                continue;
            }

            let next_pos = co.parent.and_then(|p| {
                let parent = bodies.get(p.handle)?;
                (parent.soft_ccd_prediction() > 0.0).then(|| {
                    parent.predict_position_using_velocity_and_forces_with_max_dist(
                        dt,
                        parent.soft_ccd_prediction(),
                    ) * p.pos_wrt_parent
                })
            });

            let mut aabb = co.compute_collision_aabb(0.0);
            if let Some(next_pos) = next_pos {
                let next_aabb = co.shape.compute_aabb(&next_pos).loosened(co.contact_skin());
                aabb.merge(&next_aabb);
            }
            let new_bounds = aabb_to_bounds(&aabb);

            if self.meta.get(handle.0).is_none() {
                let id = self.tree.insert_and_balance(
                    new_bounds,
                    handle,
                    ELEMENTS_PER_CELL,
                    |&handle| {
                        aabb_to_bounds(&colliders.get(handle).unwrap().compute_collision_aabb(0.0))
                    },
                );
                self.meta.insert(
                    handle.0,
                    ColliderMeta {
                        id,
                        bounds: new_bounds,
                        touching: FxHashSet::default(),
                    },
                );
            } else {
                let meta = self.meta.get_mut(handle.0).unwrap();
                let old_bounds = mem::replace(&mut meta.bounds, new_bounds);
                self.tree.update_and_balance(
                    meta.id,
                    old_bounds,
                    new_bounds,
                    ELEMENTS_PER_CELL,
                    |&handle| {
                        aabb_to_bounds(&colliders.get(handle).unwrap().compute_collision_aabb(0.0))
                    },
                );
            }
        }

        // Future work: special case initial(?) bulk inserts w/ a single balance

        for &collider1 in modified_colliders {
            let meta1 = self.meta.get_mut(collider1.0).unwrap();
            let was_touching = mem::take(&mut meta1.touching);
            let id1 = meta1.id;
            let bounds = meta1.bounds.loosened(prediction_distance as f64);
            // Detect new pairs
            for (id2, &collider2) in self.tree.intersections(bounds) {
                if id1 == id2 {
                    continue;
                }
                let meta1 = self.meta.get_mut(collider1.0).unwrap();
                meta1.touching.insert(collider2);
                if !was_touching.contains(&collider1) {
                    events.push(BroadPhasePairEvent::AddPair(ColliderPair {
                        collider1,
                        collider2,
                    }));
                    let meta2 = self.meta.get_mut(collider2.0).unwrap();
                    meta2.touching.insert(collider1);
                }
            }
            // Detect obsolete pairs
            for collider2 in was_touching {
                let meta1 = self.meta.get(collider1.0).unwrap();
                if meta1.touching.contains(&collider2) {
                    continue;
                }
                events.push(BroadPhasePairEvent::DeletePair(ColliderPair {
                    collider1,
                    collider2,
                }));
                let meta2 = self.meta.get_mut(collider2.0).unwrap();
                meta2.touching.remove(&collider1);
            }
        }
    }
}

#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
struct ColliderMeta {
    id: usize,
    bounds: Bounds,
    touching: FxHashSet<ColliderHandle>,
}

impl Default for ColliderMeta {
    fn default() -> Self {
        Self {
            id: 0,
            bounds: Bounds {
                min: Default::default(),
                max: Default::default(),
            },
            touching: FxHashSet::default(),
        }
    }
}

fn aabb_to_bounds(aabb: &Aabb) -> Bounds {
    Bounds {
        min: aabb.mins.cast::<f64>().into(),
        max: aabb.maxs.cast::<f64>().into(),
    }
}

fn sort2<T: PartialOrd>(a: T, b: T) -> (T, T) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

#[cfg(feature = "dim2")]
type SieveTree<T> = sieve_tree::SieveTree<2, 6, T>;
#[cfg(feature = "dim3")]
type SieveTree<T> = sieve_tree::SieveTree<3, 4, T>;

#[cfg(feature = "dim2")]
type Bounds = sieve_tree::Bounds<2>;
#[cfg(feature = "dim3")]
type Bounds = sieve_tree::Bounds<3>;
