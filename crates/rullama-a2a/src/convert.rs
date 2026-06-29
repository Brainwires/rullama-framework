//! Conversions between hand-written serde types and proto-generated types.
//!
//! These are gated behind the `grpc` feature and enable the gRPC service layer
//! to use the same `A2aHandler` trait as JSON-RPC and REST.

#[cfg(feature = "grpc")]
mod grpc_convert {
    use std::collections::HashMap;

    use crate::agent_card::*;
    use crate::proto::lf_a2a_v1 as pb;
    use crate::push_notification::{AuthenticationInfo, TaskPushNotificationConfig};
    use crate::task::{Task, TaskState, TaskStatus};
    use crate::types::{Artifact, Message, Part, Role};

    // ===================================================================
    // Helpers: HashMap<String, serde_json::Value> <-> prost_types::Struct
    // ===================================================================

    pub(crate) fn hashmap_to_struct(m: HashMap<String, serde_json::Value>) -> prost_types::Struct {
        prost_types::Struct {
            fields: m
                .into_iter()
                .map(|(k, v)| (k, json_to_prost_value(v)))
                .collect(),
        }
    }

    pub(crate) fn struct_to_hashmap(s: prost_types::Struct) -> HashMap<String, serde_json::Value> {
        s.fields
            .into_iter()
            .map(|(k, v)| (k, prost_value_to_json(v)))
            .collect()
    }

    fn json_to_prost_value(v: serde_json::Value) -> prost_types::Value {
        use prost_types::value::Kind;
        let kind = match v {
            serde_json::Value::Null => Kind::NullValue(0),
            serde_json::Value::Bool(b) => Kind::BoolValue(b),
            serde_json::Value::Number(n) => Kind::NumberValue(n.as_f64().unwrap_or(0.0)),
            serde_json::Value::String(s) => Kind::StringValue(s),
            serde_json::Value::Array(arr) => Kind::ListValue(prost_types::ListValue {
                values: arr.into_iter().map(json_to_prost_value).collect(),
            }),
            serde_json::Value::Object(obj) => Kind::StructValue(prost_types::Struct {
                fields: obj
                    .into_iter()
                    .map(|(k, v)| (k, json_to_prost_value(v)))
                    .collect(),
            }),
        };
        prost_types::Value { kind: Some(kind) }
    }

    fn prost_value_to_json(v: prost_types::Value) -> serde_json::Value {
        use prost_types::value::Kind;
        match v.kind {
            Some(Kind::NullValue(_)) | None => serde_json::Value::Null,
            Some(Kind::BoolValue(b)) => serde_json::Value::Bool(b),
            Some(Kind::NumberValue(n)) => serde_json::Number::from_f64(n)
                .map(serde_json::Value::Number)
                .unwrap_or(serde_json::Value::Null),
            Some(Kind::StringValue(s)) => serde_json::Value::String(s),
            Some(Kind::ListValue(l)) => {
                serde_json::Value::Array(l.values.into_iter().map(prost_value_to_json).collect())
            }
            Some(Kind::StructValue(s)) => {
                let map: serde_json::Map<String, serde_json::Value> = s
                    .fields
                    .into_iter()
                    .map(|(k, v)| (k, prost_value_to_json(v)))
                    .collect();
                serde_json::Value::Object(map)
            }
        }
    }

    fn opt_hashmap_to_struct(
        m: Option<HashMap<String, serde_json::Value>>,
    ) -> Option<prost_types::Struct> {
        m.map(hashmap_to_struct)
    }

    fn opt_struct_to_hashmap(
        s: Option<prost_types::Struct>,
    ) -> Option<HashMap<String, serde_json::Value>> {
        s.map(struct_to_hashmap)
    }

    fn opt_empty(s: &str) -> Option<String> {
        if s.is_empty() {
            None
        } else {
            Some(s.to_string())
        }
    }

    // ===================================================================
    // Role
    // ===================================================================

    impl From<Role> for i32 {
        fn from(r: Role) -> i32 {
            match r {
                Role::User => pb::Role::User as i32,
                Role::Agent => pb::Role::Agent as i32,
                Role::Unspecified => pb::Role::Unspecified as i32,
            }
        }
    }

    impl From<i32> for Role {
        fn from(v: i32) -> Self {
            match v {
                1 => Role::User,
                2 => Role::Agent,
                _ => Role::Unspecified,
            }
        }
    }

    // ===================================================================
    // TaskState
    // ===================================================================

    impl From<TaskState> for i32 {
        fn from(s: TaskState) -> i32 {
            match s {
                TaskState::Unspecified => 0,
                TaskState::Submitted => 1,
                TaskState::Working => 2,
                TaskState::Completed => 3,
                TaskState::Failed => 4,
                TaskState::Canceled => 5,
                TaskState::InputRequired => 6,
                TaskState::Rejected => 7,
                TaskState::AuthRequired => 8,
            }
        }
    }

    impl From<i32> for TaskState {
        fn from(v: i32) -> Self {
            match v {
                1 => TaskState::Submitted,
                2 => TaskState::Working,
                3 => TaskState::Completed,
                4 => TaskState::Failed,
                5 => TaskState::Canceled,
                6 => TaskState::InputRequired,
                7 => TaskState::Rejected,
                8 => TaskState::AuthRequired,
                _ => TaskState::Unspecified,
            }
        }
    }

    // ===================================================================
    // Part (flat struct with optional fields)
    // ===================================================================

    impl From<Part> for pb::Part {
        fn from(p: Part) -> pb::Part {
            // Map the flat Part struct to the proto Part with oneof content.
            let content = if let Some(text) = p.text {
                Some(pb::part::Content::Text(text))
            } else if let Some(url) = p.url {
                Some(pb::part::Content::Url(url))
            } else if let Some(raw) = p.raw {
                Some(pb::part::Content::Raw(raw.into_bytes()))
            } else {
                p.data
                    .map(|data| pb::part::Content::Data(json_to_prost_value(data)))
            };

            pb::Part {
                content,
                metadata: opt_hashmap_to_struct(p.metadata),
                filename: p.filename.unwrap_or_default(),
                media_type: p.media_type.unwrap_or_default(),
            }
        }
    }

    impl From<pb::Part> for Part {
        fn from(p: pb::Part) -> Part {
            let metadata = opt_struct_to_hashmap(p.metadata);
            let media_type = opt_empty(&p.media_type);
            let filename = opt_empty(&p.filename);

            match p.content {
                Some(pb::part::Content::Text(t)) => Part {
                    text: Some(t),
                    raw: None,
                    url: None,
                    data: None,
                    media_type,
                    filename,
                    metadata,
                },
                Some(pb::part::Content::Url(u)) => Part {
                    text: None,
                    raw: None,
                    url: Some(u),
                    data: None,
                    media_type,
                    filename,
                    metadata,
                },
                Some(pb::part::Content::Raw(b)) => Part {
                    text: None,
                    raw: Some(String::from_utf8_lossy(&b).to_string()),
                    url: None,
                    data: None,
                    media_type,
                    filename,
                    metadata,
                },
                Some(pb::part::Content::Data(v)) => Part {
                    text: None,
                    raw: None,
                    url: None,
                    data: Some(prost_value_to_json(v)),
                    media_type,
                    filename,
                    metadata,
                },
                None => Part {
                    text: Some(String::new()),
                    raw: None,
                    url: None,
                    data: None,
                    media_type,
                    filename,
                    metadata,
                },
            }
        }
    }

    // ===================================================================
    // Message (no more `kind` field)
    // ===================================================================

    impl From<Message> for pb::Message {
        fn from(m: Message) -> pb::Message {
            pb::Message {
                message_id: m.message_id,
                context_id: m.context_id.unwrap_or_default(),
                task_id: m.task_id.unwrap_or_default(),
                role: i32::from(m.role),
                parts: m.parts.into_iter().map(Into::into).collect(),
                metadata: opt_hashmap_to_struct(m.metadata),
                extensions: m.extensions.unwrap_or_default(),
                reference_task_ids: m.reference_task_ids.unwrap_or_default(),
            }
        }
    }

    impl From<pb::Message> for Message {
        fn from(m: pb::Message) -> Message {
            Message {
                message_id: m.message_id,
                role: Role::from(m.role),
                parts: m.parts.into_iter().map(Into::into).collect(),
                context_id: opt_empty(&m.context_id),
                task_id: opt_empty(&m.task_id),
                reference_task_ids: if m.reference_task_ids.is_empty() {
                    None
                } else {
                    Some(m.reference_task_ids)
                },
                metadata: opt_struct_to_hashmap(m.metadata),
                extensions: if m.extensions.is_empty() {
                    None
                } else {
                    Some(m.extensions)
                },
            }
        }
    }

    // ===================================================================
    // Artifact
    // ===================================================================

    impl From<Artifact> for pb::Artifact {
        fn from(a: Artifact) -> pb::Artifact {
            pb::Artifact {
                artifact_id: a.artifact_id,
                name: a.name.unwrap_or_default(),
                description: a.description.unwrap_or_default(),
                parts: a.parts.into_iter().map(Into::into).collect(),
                metadata: opt_hashmap_to_struct(a.metadata),
                extensions: a.extensions.unwrap_or_default(),
            }
        }
    }

    impl From<pb::Artifact> for Artifact {
        fn from(a: pb::Artifact) -> Artifact {
            Artifact {
                artifact_id: a.artifact_id,
                name: opt_empty(&a.name),
                description: opt_empty(&a.description),
                parts: a.parts.into_iter().map(Into::into).collect(),
                metadata: opt_struct_to_hashmap(a.metadata),
                extensions: if a.extensions.is_empty() {
                    None
                } else {
                    Some(a.extensions)
                },
            }
        }
    }

    // ===================================================================
    // TaskStatus
    // ===================================================================

    impl From<TaskStatus> for pb::TaskStatus {
        fn from(s: TaskStatus) -> pb::TaskStatus {
            pb::TaskStatus {
                state: i32::from(s.state),
                message: s.message.map(Into::into),
                timestamp: s.timestamp.and_then(|t| {
                    chrono::DateTime::parse_from_rfc3339(&t)
                        .ok()
                        .map(|dt| prost_types::Timestamp {
                            seconds: dt.timestamp(),
                            nanos: dt.timestamp_subsec_nanos() as i32,
                        })
                }),
            }
        }
    }

    impl From<pb::TaskStatus> for TaskStatus {
        fn from(s: pb::TaskStatus) -> TaskStatus {
            TaskStatus {
                state: TaskState::from(s.state),
                message: s.message.map(Into::into),
                timestamp: s.timestamp.and_then(|t| {
                    chrono::DateTime::from_timestamp(t.seconds, t.nanos as u32)
                        .map(|dt| dt.to_rfc3339())
                }),
            }
        }
    }

    // ===================================================================
    // Task (no more `kind` field)
    // ===================================================================

    impl From<Task> for pb::Task {
        fn from(t: Task) -> pb::Task {
            pb::Task {
                id: t.id,
                context_id: t.context_id.unwrap_or_default(),
                status: Some(t.status.into()),
                artifacts: t
                    .artifacts
                    .unwrap_or_default()
                    .into_iter()
                    .map(Into::into)
                    .collect(),
                history: t
                    .history
                    .unwrap_or_default()
                    .into_iter()
                    .map(Into::into)
                    .collect(),
                metadata: opt_hashmap_to_struct(t.metadata),
            }
        }
    }

    impl From<pb::Task> for Task {
        fn from(t: pb::Task) -> Task {
            Task {
                id: t.id,
                context_id: opt_empty(&t.context_id),
                status: t.status.map(Into::into).unwrap_or(TaskStatus {
                    state: TaskState::Unspecified,
                    message: None,
                    timestamp: None,
                }),
                artifacts: if t.artifacts.is_empty() {
                    None
                } else {
                    Some(t.artifacts.into_iter().map(Into::into).collect())
                },
                history: if t.history.is_empty() {
                    None
                } else {
                    Some(t.history.into_iter().map(Into::into).collect())
                },
                metadata: opt_struct_to_hashmap(t.metadata),
            }
        }
    }

    // ===================================================================
    // AuthenticationInfo
    // ===================================================================

    impl From<AuthenticationInfo> for pb::AuthenticationInfo {
        fn from(a: AuthenticationInfo) -> pb::AuthenticationInfo {
            pb::AuthenticationInfo {
                scheme: a.scheme,
                credentials: a.credentials.unwrap_or_default(),
            }
        }
    }

    impl From<pb::AuthenticationInfo> for AuthenticationInfo {
        fn from(a: pb::AuthenticationInfo) -> AuthenticationInfo {
            AuthenticationInfo {
                scheme: a.scheme,
                credentials: opt_empty(&a.credentials),
            }
        }
    }

    // ===================================================================
    // TaskPushNotificationConfig (id -> config_id)
    // ===================================================================

    impl From<TaskPushNotificationConfig> for pb::TaskPushNotificationConfig {
        fn from(c: TaskPushNotificationConfig) -> pb::TaskPushNotificationConfig {
            pb::TaskPushNotificationConfig {
                tenant: c.tenant.unwrap_or_default(),
                id: c.config_id.unwrap_or_default(),
                task_id: c.task_id,
                url: c.url,
                token: c.token.unwrap_or_default(),
                authentication: c.authentication.map(Into::into),
            }
        }
    }

    impl From<pb::TaskPushNotificationConfig> for TaskPushNotificationConfig {
        fn from(c: pb::TaskPushNotificationConfig) -> TaskPushNotificationConfig {
            TaskPushNotificationConfig {
                tenant: opt_empty(&c.tenant),
                config_id: opt_empty(&c.id),
                task_id: c.task_id,
                url: c.url,
                token: opt_empty(&c.token),
                authentication: c.authentication.map(Into::into),
                created_at: None,
            }
        }
    }

    // ===================================================================
    // AgentProvider
    // ===================================================================

    impl From<AgentProvider> for pb::AgentProvider {
        fn from(p: AgentProvider) -> pb::AgentProvider {
            pb::AgentProvider {
                url: p.url,
                organization: p.organization,
            }
        }
    }

    impl From<pb::AgentProvider> for AgentProvider {
        fn from(p: pb::AgentProvider) -> AgentProvider {
            AgentProvider {
                url: p.url,
                organization: p.organization,
            }
        }
    }

    // ===================================================================
    // AgentExtension
    // ===================================================================

    impl From<AgentExtension> for pb::AgentExtension {
        fn from(e: AgentExtension) -> pb::AgentExtension {
            pb::AgentExtension {
                uri: e.uri,
                description: e.description.unwrap_or_default(),
                required: e.required,
                params: e.params.map(hashmap_to_struct),
            }
        }
    }

    impl From<pb::AgentExtension> for AgentExtension {
        fn from(e: pb::AgentExtension) -> AgentExtension {
            AgentExtension {
                uri: e.uri,
                description: opt_empty(&e.description),
                required: e.required,
                params: e.params.map(struct_to_hashmap),
            }
        }
    }

    // ===================================================================
    // AgentCapabilities
    // ===================================================================

    impl From<AgentCapabilities> for pb::AgentCapabilities {
        fn from(c: AgentCapabilities) -> pb::AgentCapabilities {
            pb::AgentCapabilities {
                streaming: c.streaming,
                push_notifications: c.push_notifications,
                extended_agent_card: c.extended_agent_card,
                extensions: c
                    .extensions
                    .unwrap_or_default()
                    .into_iter()
                    .map(Into::into)
                    .collect(),
            }
        }
    }

    impl From<pb::AgentCapabilities> for AgentCapabilities {
        fn from(c: pb::AgentCapabilities) -> AgentCapabilities {
            AgentCapabilities {
                streaming: c.streaming,
                push_notifications: c.push_notifications,
                extended_agent_card: c.extended_agent_card,
                extensions: if c.extensions.is_empty() {
                    None
                } else {
                    Some(c.extensions.into_iter().map(Into::into).collect())
                },
            }
        }
    }

    // ===================================================================
    // AgentSkill
    // ===================================================================

    impl From<AgentSkill> for pb::AgentSkill {
        fn from(s: AgentSkill) -> pb::AgentSkill {
            pb::AgentSkill {
                id: s.id,
                name: s.name,
                description: s.description,
                tags: s.tags,
                examples: s.examples.unwrap_or_default(),
                input_modes: s.input_modes.unwrap_or_default(),
                output_modes: s.output_modes.unwrap_or_default(),
                security_requirements: s
                    .security_requirements
                    .unwrap_or_default()
                    .into_iter()
                    .map(Into::into)
                    .collect(),
            }
        }
    }

    impl From<pb::AgentSkill> for AgentSkill {
        fn from(s: pb::AgentSkill) -> AgentSkill {
            AgentSkill {
                id: s.id,
                name: s.name,
                description: s.description,
                tags: s.tags,
                examples: if s.examples.is_empty() {
                    None
                } else {
                    Some(s.examples)
                },
                input_modes: if s.input_modes.is_empty() {
                    None
                } else {
                    Some(s.input_modes)
                },
                output_modes: if s.output_modes.is_empty() {
                    None
                } else {
                    Some(s.output_modes)
                },
                security_requirements: if s.security_requirements.is_empty() {
                    None
                } else {
                    Some(
                        s.security_requirements
                            .into_iter()
                            .map(Into::into)
                            .collect(),
                    )
                },
            }
        }
    }

    // ===================================================================
    // AgentInterface
    // ===================================================================

    impl From<AgentInterface> for pb::AgentInterface {
        fn from(i: AgentInterface) -> pb::AgentInterface {
            pb::AgentInterface {
                url: i.url,
                protocol_binding: i.protocol_binding,
                tenant: i.tenant.unwrap_or_default(),
                protocol_version: i.protocol_version,
            }
        }
    }

    impl From<pb::AgentInterface> for AgentInterface {
        fn from(i: pb::AgentInterface) -> AgentInterface {
            AgentInterface {
                url: i.url,
                protocol_binding: i.protocol_binding,
                tenant: opt_empty(&i.tenant),
                protocol_version: i.protocol_version,
            }
        }
    }

    // ===================================================================
    // AgentCardSignature
    // ===================================================================

    impl From<AgentCardSignature> for pb::AgentCardSignature {
        fn from(s: AgentCardSignature) -> pb::AgentCardSignature {
            pb::AgentCardSignature {
                protected: s.protected,
                signature: s.signature,
                header: s.header.map(hashmap_to_struct),
            }
        }
    }

    impl From<pb::AgentCardSignature> for AgentCardSignature {
        fn from(s: pb::AgentCardSignature) -> AgentCardSignature {
            AgentCardSignature {
                protected: s.protected,
                signature: s.signature,
                header: s.header.map(struct_to_hashmap),
            }
        }
    }

    // ===================================================================
    // SecurityRequirement
    // ===================================================================

    impl From<SecurityRequirement> for pb::SecurityRequirement {
        fn from(r: SecurityRequirement) -> pb::SecurityRequirement {
            pb::SecurityRequirement {
                schemes: r
                    .schemes
                    .into_iter()
                    .map(|(k, v)| (k, pb::StringList { list: v }))
                    .collect(),
            }
        }
    }

    impl From<pb::SecurityRequirement> for SecurityRequirement {
        fn from(r: pb::SecurityRequirement) -> SecurityRequirement {
            SecurityRequirement {
                schemes: r.schemes.into_iter().map(|(k, v)| (k, v.list)).collect(),
            }
        }
    }

    // ===================================================================
    // SecurityScheme (now a struct with optional wrapper fields)
    // ===================================================================

    impl From<SecurityScheme> for pb::SecurityScheme {
        fn from(s: SecurityScheme) -> pb::SecurityScheme {
            let scheme = if let Some(api_key) = s.api_key {
                Some(pb::security_scheme::Scheme::ApiKeySecurityScheme(
                    pb::ApiKeySecurityScheme {
                        description: api_key.description.unwrap_or_default(),
                        location: api_key.location,
                        name: api_key.name,
                    },
                ))
            } else if let Some(http_auth) = s.http_auth {
                Some(pb::security_scheme::Scheme::HttpAuthSecurityScheme(
                    pb::HttpAuthSecurityScheme {
                        description: http_auth.description.unwrap_or_default(),
                        scheme: http_auth.scheme,
                        bearer_format: http_auth.bearer_format.unwrap_or_default(),
                    },
                ))
            } else if let Some(oauth2) = s.oauth2 {
                Some(pb::security_scheme::Scheme::Oauth2SecurityScheme(
                    pb::OAuth2SecurityScheme {
                        description: oauth2.description.unwrap_or_default(),
                        flows: Some(oauth2.flows.into()),
                        oauth2_metadata_url: oauth2.oauth2_metadata_url.unwrap_or_default(),
                    },
                ))
            } else if let Some(oidc) = s.open_id_connect {
                Some(pb::security_scheme::Scheme::OpenIdConnectSecurityScheme(
                    pb::OpenIdConnectSecurityScheme {
                        description: oidc.description.unwrap_or_default(),
                        open_id_connect_url: oidc.open_id_connect_url,
                    },
                ))
            } else if let Some(mtls) = s.mtls {
                Some(pb::security_scheme::Scheme::MtlsSecurityScheme(
                    pb::MutualTlsSecurityScheme {
                        description: mtls.description.unwrap_or_default(),
                    },
                ))
            } else {
                None
            };
            pb::SecurityScheme { scheme }
        }
    }

    impl From<pb::SecurityScheme> for SecurityScheme {
        fn from(s: pb::SecurityScheme) -> SecurityScheme {
            match s.scheme {
                Some(pb::security_scheme::Scheme::ApiKeySecurityScheme(a)) => SecurityScheme {
                    api_key: Some(ApiKeySecurityScheme {
                        name: a.name,
                        location: a.location,
                        description: opt_empty(&a.description),
                    }),
                    http_auth: None,
                    oauth2: None,
                    open_id_connect: None,
                    mtls: None,
                },
                Some(pb::security_scheme::Scheme::HttpAuthSecurityScheme(h)) => SecurityScheme {
                    api_key: None,
                    http_auth: Some(HttpAuthSecurityScheme {
                        scheme: h.scheme,
                        bearer_format: opt_empty(&h.bearer_format),
                        description: opt_empty(&h.description),
                    }),
                    oauth2: None,
                    open_id_connect: None,
                    mtls: None,
                },
                Some(pb::security_scheme::Scheme::Oauth2SecurityScheme(o)) => SecurityScheme {
                    api_key: None,
                    http_auth: None,
                    oauth2: Some(OAuth2SecurityScheme {
                        flows: o.flows.map(Into::into).unwrap_or(OAuthFlows {
                            authorization_code: None,
                            client_credentials: Some(ClientCredentialsOAuthFlow {
                                token_url: String::new(),
                                refresh_url: None,
                                scopes: HashMap::new(),
                            }),
                            implicit: None,
                            password: None,
                            device_code: None,
                        }),
                        description: opt_empty(&o.description),
                        oauth2_metadata_url: opt_empty(&o.oauth2_metadata_url),
                    }),
                    open_id_connect: None,
                    mtls: None,
                },
                Some(pb::security_scheme::Scheme::OpenIdConnectSecurityScheme(o)) => {
                    SecurityScheme {
                        api_key: None,
                        http_auth: None,
                        oauth2: None,
                        open_id_connect: Some(OpenIdConnectSecurityScheme {
                            open_id_connect_url: o.open_id_connect_url,
                            description: opt_empty(&o.description),
                        }),
                        mtls: None,
                    }
                }
                Some(pb::security_scheme::Scheme::MtlsSecurityScheme(m)) => SecurityScheme {
                    api_key: None,
                    http_auth: None,
                    oauth2: None,
                    open_id_connect: None,
                    mtls: Some(MutualTlsSecurityScheme {
                        description: opt_empty(&m.description),
                    }),
                },
                None => SecurityScheme {
                    api_key: None,
                    http_auth: None,
                    oauth2: None,
                    open_id_connect: None,
                    mtls: None,
                },
            }
        }
    }

    // ===================================================================
    // OAuthFlows (now a struct with optional wrapper fields)
    // ===================================================================

    impl From<OAuthFlows> for pb::OAuthFlows {
        fn from(f: OAuthFlows) -> pb::OAuthFlows {
            let flow = if let Some(ac) = f.authorization_code {
                Some(pb::o_auth_flows::Flow::AuthorizationCode(
                    pb::AuthorizationCodeOAuthFlow {
                        authorization_url: ac.authorization_url,
                        token_url: ac.token_url,
                        refresh_url: ac.refresh_url.unwrap_or_default(),
                        scopes: ac.scopes,
                        pkce_required: ac.pkce_required.unwrap_or(false),
                    },
                ))
            } else if let Some(cc) = f.client_credentials {
                Some(pb::o_auth_flows::Flow::ClientCredentials(
                    pb::ClientCredentialsOAuthFlow {
                        token_url: cc.token_url,
                        refresh_url: cc.refresh_url.unwrap_or_default(),
                        scopes: cc.scopes,
                    },
                ))
            } else if let Some(imp) = f.implicit {
                Some(pb::o_auth_flows::Flow::Implicit(pb::ImplicitOAuthFlow {
                    authorization_url: imp.authorization_url.unwrap_or_default(),
                    refresh_url: imp.refresh_url.unwrap_or_default(),
                    scopes: imp.scopes,
                }))
            } else if let Some(pw) = f.password {
                Some(pb::o_auth_flows::Flow::Password(pb::PasswordOAuthFlow {
                    token_url: pw.token_url.unwrap_or_default(),
                    refresh_url: pw.refresh_url.unwrap_or_default(),
                    scopes: pw.scopes,
                }))
            } else if let Some(dc) = f.device_code {
                Some(pb::o_auth_flows::Flow::DeviceCode(
                    pb::DeviceCodeOAuthFlow {
                        device_authorization_url: dc.device_authorization_url,
                        token_url: dc.token_url,
                        refresh_url: dc.refresh_url.unwrap_or_default(),
                        scopes: dc.scopes,
                    },
                ))
            } else {
                None
            };
            pb::OAuthFlows { flow }
        }
    }

    impl From<pb::OAuthFlows> for OAuthFlows {
        fn from(f: pb::OAuthFlows) -> OAuthFlows {
            match f.flow {
                Some(pb::o_auth_flows::Flow::AuthorizationCode(a)) => OAuthFlows {
                    authorization_code: Some(AuthorizationCodeOAuthFlow {
                        authorization_url: a.authorization_url,
                        token_url: a.token_url,
                        refresh_url: opt_empty(&a.refresh_url),
                        scopes: a.scopes,
                        pkce_required: Some(a.pkce_required),
                    }),
                    client_credentials: None,
                    implicit: None,
                    password: None,
                    device_code: None,
                },
                Some(pb::o_auth_flows::Flow::ClientCredentials(c)) => OAuthFlows {
                    authorization_code: None,
                    client_credentials: Some(ClientCredentialsOAuthFlow {
                        token_url: c.token_url,
                        refresh_url: opt_empty(&c.refresh_url),
                        scopes: c.scopes,
                    }),
                    implicit: None,
                    password: None,
                    device_code: None,
                },
                #[allow(deprecated)]
                Some(pb::o_auth_flows::Flow::Implicit(i)) => OAuthFlows {
                    authorization_code: None,
                    client_credentials: None,
                    implicit: Some(ImplicitOAuthFlow {
                        authorization_url: opt_empty(&i.authorization_url),
                        refresh_url: opt_empty(&i.refresh_url),
                        scopes: i.scopes,
                    }),
                    password: None,
                    device_code: None,
                },
                #[allow(deprecated)]
                Some(pb::o_auth_flows::Flow::Password(p)) => OAuthFlows {
                    authorization_code: None,
                    client_credentials: None,
                    implicit: None,
                    password: Some(PasswordOAuthFlow {
                        token_url: opt_empty(&p.token_url),
                        refresh_url: opt_empty(&p.refresh_url),
                        scopes: p.scopes,
                    }),
                    device_code: None,
                },
                Some(pb::o_auth_flows::Flow::DeviceCode(d)) => OAuthFlows {
                    authorization_code: None,
                    client_credentials: None,
                    implicit: None,
                    password: None,
                    device_code: Some(DeviceCodeOAuthFlow {
                        device_authorization_url: d.device_authorization_url,
                        token_url: d.token_url,
                        refresh_url: opt_empty(&d.refresh_url),
                        scopes: d.scopes,
                    }),
                },
                None => OAuthFlows {
                    authorization_code: None,
                    client_credentials: None,
                    implicit: None,
                    password: None,
                    device_code: None,
                },
            }
        }
    }

    // ===================================================================
    // AgentCard (supported_interfaces is now Vec, not Option<Vec>)
    // ===================================================================

    impl From<AgentCard> for pb::AgentCard {
        fn from(c: AgentCard) -> pb::AgentCard {
            pb::AgentCard {
                name: c.name,
                description: c.description,
                version: c.version,
                supported_interfaces: c.supported_interfaces.into_iter().map(Into::into).collect(),
                provider: c.provider.map(Into::into),
                capabilities: Some(c.capabilities.into()),
                security_schemes: c
                    .security_schemes
                    .unwrap_or_default()
                    .into_iter()
                    .map(|(k, v)| (k, v.into()))
                    .collect(),
                security_requirements: c
                    .security_requirements
                    .unwrap_or_default()
                    .into_iter()
                    .map(Into::into)
                    .collect(),
                default_input_modes: c.default_input_modes,
                default_output_modes: c.default_output_modes,
                skills: c.skills.into_iter().map(Into::into).collect(),
                signatures: c
                    .signatures
                    .unwrap_or_default()
                    .into_iter()
                    .map(Into::into)
                    .collect(),
                documentation_url: c.documentation_url,
                icon_url: c.icon_url,
            }
        }
    }

    impl From<pb::AgentCard> for AgentCard {
        fn from(c: pb::AgentCard) -> AgentCard {
            AgentCard {
                name: c.name,
                description: c.description,
                version: c.version,
                supported_interfaces: c.supported_interfaces.into_iter().map(Into::into).collect(),
                provider: c.provider.map(Into::into),
                capabilities: c.capabilities.map(Into::into).unwrap_or_default(),
                security_schemes: {
                    let m: HashMap<String, SecurityScheme> = c
                        .security_schemes
                        .into_iter()
                        .map(|(k, v)| (k, v.into()))
                        .collect();
                    if m.is_empty() { None } else { Some(m) }
                },
                security_requirements: if c.security_requirements.is_empty() {
                    None
                } else {
                    Some(
                        c.security_requirements
                            .into_iter()
                            .map(Into::into)
                            .collect(),
                    )
                },
                default_input_modes: c.default_input_modes,
                default_output_modes: c.default_output_modes,
                skills: c.skills.into_iter().map(Into::into).collect(),
                signatures: if c.signatures.is_empty() {
                    None
                } else {
                    Some(c.signatures.into_iter().map(Into::into).collect())
                },
                documentation_url: c.documentation_url,
                icon_url: c.icon_url,
            }
        }
    }
}
