/// Access Control List (ACL) for the Matter data model.
///
/// Enforces privilege-based access to endpoints and clusters per Matter spec §6.6.
use super::Privilege;

// ── AclTarget ─────────────────────────────────────────────────────────────────

/// A target in an ACL entry — identifies a cluster and/or endpoint.
///
/// `None` in either field means "any".
#[derive(Debug, Clone)]
pub struct AclTarget {
    /// Cluster identifier.  `None` = all clusters.
    pub cluster: Option<u32>,
    /// Endpoint identifier.  `None` = all endpoints.
    pub endpoint: Option<u16>,
}

// ── AccessControlEntry ────────────────────────────────────────────────────────

/// A single ACL entry granting a privilege to a set of subjects on a set of targets.
#[derive(Debug, Clone)]
pub struct AccessControlEntry {
    /// Fabric this entry belongs to.
    pub fabric_index: u8,
    /// Privilege granted.
    pub privilege: Privilege,
    /// Authentication mode: 1 = PASE, 2 = CASE, 3 = Group.
    pub auth_mode: u8,
    /// Subject node IDs granted this entry.  Empty = any subject.
    pub subjects: Vec<u64>,
    /// Targets restricted by this entry.  Empty = all clusters/endpoints.
    pub targets: Vec<AclTarget>,
}

// ── AccessControlList ─────────────────────────────────────────────────────────

/// The device-wide Access Control List.
pub struct AccessControlList {
    entries: Vec<AccessControlEntry>,
}

impl AccessControlList {
    /// Create an empty ACL.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Append an entry to the list.
    pub fn add_entry(&mut self, entry: AccessControlEntry) {
        self.entries.push(entry);
    }

    /// Return `true` if `subject` on fabric `fabric_index` has at least
    /// `required` privilege on (`endpoint`, `cluster`).
    pub fn check_access(
        &self,
        fabric_index: u8,
        subject: u64,
        endpoint: u16,
        cluster: u32,
        required: Privilege,
    ) -> bool {
        for entry in &self.entries {
            // Fabric must match.
            if entry.fabric_index != fabric_index {
                continue;
            }
            // The granted privilege must be sufficient.
            if entry.privilege < required {
                continue;
            }
            // Subject check: empty subjects list means any.
            if !entry.subjects.is_empty() && !entry.subjects.contains(&subject) {
                continue;
            }
            // Target check: empty targets list means all endpoints/clusters.
            if !entry.targets.is_empty() {
                let target_match = entry.targets.iter().any(|t| {
                    let ep_ok = t.endpoint.is_none_or(|e| e == endpoint);
                    let cl_ok = t.cluster.is_none_or(|c| c == cluster);
                    ep_ok && cl_ok
                });
                if !target_match {
                    continue;
                }
            }
            return true;
        }
        false
    }

    /// Bootstrap grant: fabric 0 gets `Administer` privilege on all
    /// endpoints/clusters with no subject restriction.  Used during
    /// commissioning before a real NOC is installed.
    pub fn grant_commissioning_access(&mut self) {
        self.entries.push(AccessControlEntry {
            fabric_index: 0,
            privilege: Privilege::Administer,
            auth_mode: 1, // PASE
            subjects: vec![],
            targets: vec![],
        });
    }
}

impl Default for AccessControlList {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acl_commissioning_access_grants_administer() {
        let mut acl = AccessControlList::new();
        acl.grant_commissioning_access();

        // Any subject on fabric 0 gets Administer on any endpoint/cluster.
        assert!(acl.check_access(0, 0xDEAD_BEEF_0000_0001, 0, 0x0028, Privilege::Administer));
        assert!(acl.check_access(0, 0, 1, 0x003E, Privilege::View));
        // Fabric 1 has nothing yet.
        assert!(!acl.check_access(1, 0, 0, 0x0028, Privilege::View));
    }

    #[test]
    fn acl_specific_subject_check() {
        let mut acl = AccessControlList::new();
        let node_a: u64 = 0x0000_0000_0000_0001;
        let node_b: u64 = 0x0000_0000_0000_0002;

        acl.add_entry(AccessControlEntry {
            fabric_index: 1,
            privilege: Privilege::Operate,
            auth_mode: 2, // CASE
            subjects: vec![node_a],
            targets: vec![AclTarget {
                cluster: Some(0x0006),
                endpoint: Some(1),
            }],
        });

        // node_a has Operate on ep=1, cluster=0x0006.
        assert!(acl.check_access(1, node_a, 1, 0x0006, Privilege::View));
        assert!(acl.check_access(1, node_a, 1, 0x0006, Privilege::Operate));
        // node_a does NOT have Manage.
        assert!(!acl.check_access(1, node_a, 1, 0x0006, Privilege::Manage));
        // node_b is not in the subjects list.
        assert!(!acl.check_access(1, node_b, 1, 0x0006, Privilege::View));
        // Different endpoint.
        assert!(!acl.check_access(1, node_a, 2, 0x0006, Privilege::View));
        // Different cluster.
        assert!(!acl.check_access(1, node_a, 1, 0x0008, Privilege::View));
    }
}
