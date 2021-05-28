use crate::dynamics::{
    RigidBodyCcd, RigidBodyHandle, RigidBodyMassProps, RigidBodyPosition, RigidBodyVelocity,
};
use crate::geometry::{
    ColliderHandle, ColliderParent, ColliderPosition, ColliderShape, ColliderType,
};
use crate::math::Real;
use parry::query::{NonlinearRigidMotion, QueryDispatcher, Unsupported};

#[derive(Copy, Clone, Debug)]
pub struct TOIEntry {
    pub toi: Real,
    pub c1: ColliderHandle,
    pub b1: Option<RigidBodyHandle>,
    pub c2: ColliderHandle,
    pub b2: Option<RigidBodyHandle>,
    pub is_intersection_test: bool,
    pub timestamp: usize,
}

impl TOIEntry {
    fn new(
        toi: Real,
        c1: ColliderHandle,
        b1: Option<RigidBodyHandle>,
        c2: ColliderHandle,
        b2: Option<RigidBodyHandle>,
        is_intersection_test: bool,
        timestamp: usize,
    ) -> Self {
        Self {
            toi,
            c1,
            b1,
            c2,
            b2,
            is_intersection_test,
            timestamp,
        }
    }

    pub fn try_from_colliders<QD: ?Sized + QueryDispatcher>(
        query_dispatcher: &QD,
        ch1: ColliderHandle,
        ch2: ColliderHandle,
        c1: (
            &ColliderType,
            &ColliderShape,
            &ColliderPosition,
            Option<&ColliderParent>,
        ),
        c2: (
            &ColliderType,
            &ColliderShape,
            &ColliderPosition,
            Option<&ColliderParent>,
        ),
        b1: Option<(
            &RigidBodyPosition,
            &RigidBodyVelocity,
            &RigidBodyMassProps,
            &RigidBodyCcd,
        )>,
        b2: Option<(
            &RigidBodyPosition,
            &RigidBodyVelocity,
            &RigidBodyMassProps,
            &RigidBodyCcd,
        )>,
        frozen1: Option<Real>,
        frozen2: Option<Real>,
        start_time: Real,
        end_time: Real,
        smallest_contact_dist: Real,
    ) -> Option<Self> {
        assert!(start_time <= end_time);
        if b1.is_none() && b2.is_none() {
            return None;
        }

        let (co_type1, co_shape1, co_pos1, co_parent1) = c1;
        let (co_type2, co_shape2, co_pos2, co_parent2) = c2;

        let linvel1 =
            frozen1.is_none() as u32 as Real * b1.map(|b| b.1.linvel).unwrap_or(na::zero());
        let linvel2 =
            frozen2.is_none() as u32 as Real * b2.map(|b| b.1.linvel).unwrap_or(na::zero());
        let angvel1 =
            frozen1.is_none() as u32 as Real * b1.map(|b| b.1.angvel).unwrap_or(na::zero());
        let angvel2 =
            frozen2.is_none() as u32 as Real * b2.map(|b| b.1.angvel).unwrap_or(na::zero());

        #[cfg(feature = "dim2")]
        let vel12 = (linvel2 - linvel1).norm()
            + angvel1.abs() * b1.map(|b| b.3.ccd_max_dist).unwrap_or(0.0)
            + angvel2.abs() * b2.map(|b| b.3.ccd_max_dist).unwrap_or(0.0);
        #[cfg(feature = "dim3")]
        let vel12 = (linvel2 - linvel1).norm()
            + angvel1.norm() * b1.map(|b| b.3.ccd_max_dist).unwrap_or(0.0)
            + angvel2.norm() * b2.map(|b| b.3.ccd_max_dist).unwrap_or(0.0);

        // We may be slightly over-conservative by taking the `max(0.0)` here.
        // But removing the `max` doesn't really affect performances so let's
        // keep it since more conservatism is good at this stage.
        let thickness = (co_shape1.0.ccd_thickness() + co_shape2.0.ccd_thickness())
            + smallest_contact_dist.max(0.0);
        let is_intersection_test = co_type1.is_sensor() || co_type2.is_sensor();

        if (end_time - start_time) * vel12 < thickness {
            return None;
        }

        // Compute the TOI.
        let identity = NonlinearRigidMotion::identity();
        let mut motion1 = b1.map(Self::body_motion).unwrap_or(identity);
        let mut motion2 = b2.map(Self::body_motion).unwrap_or(identity);

        if let Some(t) = frozen1 {
            motion1.freeze(t);
        }

        if let Some(t) = frozen2 {
            motion2.freeze(t);
        }

        let motion_c1 = motion1.prepend(co_parent1.map(|p| p.pos_wrt_parent).unwrap_or(co_pos1.0));
        let motion_c2 = motion2.prepend(co_parent2.map(|p| p.pos_wrt_parent).unwrap_or(co_pos2.0));

        // println!("start_time: {}", start_time);

        // If this is just an intersection test (i.e. with sensors)
        // then we can stop the TOI search immediately if it starts with
        // a penetration because we don't care about the whether the velocity
        // at the impact is a separating velocity or not.
        // If the TOI search involves two non-sensor colliders then
        // we don't want to stop the TOI search at the first penetration
        // because the colliders may be in a separating trajectory.
        let stop_at_penetration = is_intersection_test;

        let res_toi = query_dispatcher
            .nonlinear_time_of_impact(
                &motion_c1,
                co_shape1.as_ref(),
                &motion_c2,
                co_shape2.as_ref(),
                start_time,
                end_time,
                stop_at_penetration,
            )
            .or_else(|Unsupported| {
                // Fall back on linear TOI if nonlinear TOI isn't supported
                let pos12 = motion_c1.position_at_time(start_time).inverse()
                    * motion_c2.position_at_time(start_time);
                query_dispatcher.time_of_impact(
                    &pos12,
                    &(linvel2 - linvel1),
                    co_shape1.as_ref(),
                    co_shape2.as_ref(),
                    end_time - start_time,
                )
            })
            .ok();

        let toi = res_toi??;

        Some(Self::new(
            toi.toi,
            ch1,
            co_parent1.map(|p| p.handle),
            ch2,
            co_parent2.map(|p| p.handle),
            is_intersection_test,
            0,
        ))
    }

    fn body_motion(
        (poss, vels, mprops, ccd): (
            &RigidBodyPosition,
            &RigidBodyVelocity,
            &RigidBodyMassProps,
            &RigidBodyCcd,
        ),
    ) -> NonlinearRigidMotion {
        if ccd.ccd_active {
            NonlinearRigidMotion::new(
                poss.position,
                mprops.mass_properties.local_com,
                vels.linvel,
                vels.angvel,
            )
        } else {
            NonlinearRigidMotion::constant_position(poss.next_position)
        }
    }
}

impl PartialOrd for TOIEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        (-self.toi).partial_cmp(&(-other.toi))
    }
}

impl Ord for TOIEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl PartialEq for TOIEntry {
    fn eq(&self, other: &Self) -> bool {
        self.toi == other.toi
    }
}

impl Eq for TOIEntry {}
