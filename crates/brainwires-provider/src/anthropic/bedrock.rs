//! Amazon Bedrock auth -- AWS SigV4 request signing.
//!
//! Feature-gated behind `bedrock`.

use anyhow::Result;

/// Bedrock endpoint pattern:
/// `POST https://bedrock-runtime.{region}.amazonaws.com/model/{model_id}/invoke`
pub fn bedrock_invoke_url(region: &str, model_id: &str) -> String {
    format!(
        "https://bedrock-runtime.{region}.amazonaws.com/model/{model_id}/invoke",
        region = region,
        model_id = model_id,
    )
}

/// Bedrock streaming endpoint:
/// `POST https://bedrock-runtime.{region}.amazonaws.com/model/{model_id}/invoke-with-response-stream`
pub fn bedrock_stream_url(region: &str, model_id: &str) -> String {
    format!(
        "https://bedrock-runtime.{region}.amazonaws.com/model/{model_id}/invoke-with-response-stream",
        region = region,
        model_id = model_id,
    )
}

/// AWS SigV4 authentication for Bedrock requests.
pub struct BedrockAuth {
    region: String,
    access_key: String,
    secret_key: String,
    session_token: Option<String>,
}

impl BedrockAuth {
    /// Create from explicit credentials.
    pub fn new(
        region: String,
        access_key: String,
        secret_key: String,
        session_token: Option<String>,
    ) -> Self {
        Self {
            region,
            access_key,
            secret_key,
            session_token,
        }
    }

    /// Create from standard AWS environment variables.
    ///
    /// Reads `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`,
    /// and `AWS_DEFAULT_REGION` (defaults to `us-east-1`).
    pub fn from_environment(region_override: Option<String>) -> anyhow::Result<Self> {
        let access_key = std::env::var("AWS_ACCESS_KEY_ID").map_err(|_| {
            anyhow::anyhow!(
                "AWS_ACCESS_KEY_ID not set. Configure AWS credentials for Bedrock access."
            )
        })?;
        let secret_key = std::env::var("AWS_SECRET_ACCESS_KEY").map_err(|_| {
            anyhow::anyhow!(
                "AWS_SECRET_ACCESS_KEY not set. Configure AWS credentials for Bedrock access."
            )
        })?;
        let session_token = std::env::var("AWS_SESSION_TOKEN").ok();
        let region = region_override
            .or_else(|| std::env::var("AWS_DEFAULT_REGION").ok())
            .unwrap_or_else(|| "us-east-1".to_string());

        Ok(Self {
            region,
            access_key,
            secret_key,
            session_token,
        })
    }

    /// The AWS region for this auth context.
    pub fn region(&self) -> &str {
        &self.region
    }

    /// Sign a reqwest::Request with SigV4 for the `bedrock` service.
    ///
    /// AWS SigV4 operates on `http::Request` types. This method extracts
    /// headers from the reqwest request, signs them, then applies the
    /// resulting auth headers back onto the reqwest request.
    pub async fn sign_request(&self, request: &mut reqwest::Request) -> Result<()> {
        use aws_credential_types::Credentials;
        use aws_sigv4::http_request::{SignableBody, SignableRequest, SigningSettings, sign};
        use aws_sigv4::sign::v4;
        use std::time::SystemTime;

        let credentials = Credentials::new(
            &self.access_key,
            &self.secret_key,
            self.session_token.clone(),
            None, // expiry
            "brainwires-bedrock",
        );

        let settings = SigningSettings::default();
        let identity = credentials.into();
        let signing_params = v4::SigningParams::builder()
            .identity(&identity)
            .region(&self.region)
            .name("bedrock")
            .time(SystemTime::now())
            .settings(settings)
            .build()?
            .into();

        let signable_request = SignableRequest::new(
            request.method().as_str(),
            request.url().as_str(),
            request
                .headers()
                .iter()
                .map(|(k, v)| (k.as_str(), std::str::from_utf8(v.as_bytes()).unwrap_or(""))),
            SignableBody::Bytes(request.body().and_then(|b| b.as_bytes()).unwrap_or(&[])),
        )?;

        let (signing_instructions, _signature) =
            sign(signable_request, &signing_params)?.into_parts();

        // Build a temporary http::Request to apply signing instructions,
        // then copy the resulting headers back onto the reqwest request.
        let mut tmp = http::Request::builder()
            .method(request.method().as_str())
            .uri(request.url().as_str())
            .body(())
            .expect("valid request parts");
        *tmp.headers_mut() = request.headers().clone();
        signing_instructions.apply_to_request_http1x(&mut tmp);
        *request.headers_mut() = tmp.into_parts().0.headers;

        // Add Anthropic version header for Bedrock
        request.headers_mut().insert(
            "anthropic_version",
            "bedrock-2023-05-31".parse().expect("valid header value"),
        );

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bedrock_invoke_url_includes_region_and_model() {
        let url = bedrock_invoke_url("us-east-1", "anthropic.claude-sonnet-4-6-v1:0");
        assert!(url.contains("us-east-1"));
        assert!(url.contains("anthropic.claude-sonnet-4-6-v1:0"));
        assert!(url.starts_with("https://bedrock-runtime."));
        assert!(url.ends_with("/invoke"));
    }

    #[test]
    fn bedrock_stream_url_includes_invoke_with_response_stream() {
        let url = bedrock_stream_url("eu-west-1", "anthropic.claude-instant-v1");
        assert!(url.contains("eu-west-1"));
        assert!(url.contains("invoke-with-response-stream"));
    }

    #[test]
    fn bedrock_auth_new_stores_credentials() {
        let auth = BedrockAuth::new(
            "us-west-2".to_string(),
            "AKID".to_string(),
            "SECRET".to_string(),
            None,
        );
        assert_eq!(auth.region(), "us-west-2");
    }

    #[test]
    fn bedrock_auth_from_env_fails_without_credentials() {
        // Make sure no AWS env vars are set (or they are set to garbage)
        // This should fail because AWS_ACCESS_KEY_ID won't be set in CI.
        // SAFETY: this is a single-threaded unit test; we mutate process env
        // before calling into BedrockAuth, with no other threads racing.
        unsafe {
            std::env::remove_var("AWS_ACCESS_KEY_ID");
        }
        let result = BedrockAuth::from_environment(None);
        assert!(result.is_err());
    }
}
