//! Prompt Generation with SEAL+BKS+PKS Integration
//!
//! This module generates dynamic prompts using SEAL quality scores,
//! BKS shared knowledge, and PKS user preferences.

use super::clustering::TaskClusterManager;
use super::library::TechniqueLibrary;
use super::techniques::{
    ComplexityLevel, PromptingTechnique, TechniqueCategory, TechniqueMetadata,
};
use crate::seal::SealProcessingResult;
use anyhow::{Result, anyhow};
use brainwires_knowledge::knowledge::bks_pks::{BehavioralKnowledgeCache, PersonalKnowledgeCache};
use std::sync::Arc;
use tokio::sync::Mutex;

/// Generates optimized prompts based on task characteristics
pub struct PromptGenerator {
    library: TechniqueLibrary,
    cluster_manager: TaskClusterManager,
    bks_cache: Option<Arc<Mutex<BehavioralKnowledgeCache>>>,
    pks_cache: Option<Arc<Mutex<PersonalKnowledgeCache>>>,
}

impl PromptGenerator {
    /// Create a new prompt generator
    pub fn new(library: TechniqueLibrary, cluster_manager: TaskClusterManager) -> Self {
        Self {
            library,
            cluster_manager,
            bks_cache: None,
            pks_cache: None,
        }
    }

    /// Set knowledge caches for BKS/PKS integration
    pub fn with_knowledge(
        mut self,
        bks: Arc<Mutex<BehavioralKnowledgeCache>>,
        pks: Arc<Mutex<PersonalKnowledgeCache>>,
    ) -> Self {
        self.bks_cache = Some(bks);
        self.pks_cache = Some(pks);
        self
    }

    /// Generate optimized prompt (SEAL+BKS+PKS-informed)
    ///
    /// This is the main entry point that orchestrates:
    /// 1. Task classification using cluster matching
    /// 2. Multi-source technique selection (PKS > BKS > cluster default)
    /// 3. SEAL quality filtering
    /// 4. Dynamic prompt composition
    ///
    /// # Arguments
    /// * `task_description` - The task to generate a prompt for
    /// * `task_embedding` - Pre-computed embedding of the task
    /// * `seal_result` - Optional SEAL processing result for enhancement
    ///
    /// # Returns
    /// * Generated system prompt string
    pub async fn generate_prompt(
        &self,
        task_description: &str,
        task_embedding: &[f32],
        seal_result: Option<&SealProcessingResult>,
    ) -> Result<GeneratedPrompt> {
        // Step 1: Find matching cluster (SEAL-enhanced)
        let (cluster, similarity) = self
            .cluster_manager
            .find_matching_cluster(task_embedding, seal_result)?;

        // Step 2: Get SEAL quality score
        let seal_quality = seal_result.map(|r| r.quality_score).unwrap_or(0.5);

        // Step 3: Select techniques (multi-source)
        let techniques = self
            .select_techniques_multi_source(cluster.id.as_str(), seal_quality, seal_result)
            .await?;

        // Step 4: Generate prompt from techniques
        let prompt_text =
            self.compose_prompt(task_description, &techniques, cluster.description.as_str());

        Ok(GeneratedPrompt {
            system_prompt: prompt_text,
            cluster_id: cluster.id.clone(),
            techniques: techniques.iter().map(|t| t.technique.clone()).collect(),
            seal_quality,
            similarity_score: similarity,
        })
    }

    /// Select techniques from multiple sources (cluster + BKS + PKS + SEAL quality)
    ///
    /// Priority order: PKS (user preference) > BKS (collective learning) > cluster default
    async fn select_techniques_multi_source<'a>(
        &'a self,
        cluster_id: &str,
        seal_quality: f32,
        _seal_result: Option<&SealProcessingResult>,
    ) -> Result<Vec<&'a TechniqueMetadata>> {
        // Source 1: Get cluster's default techniques
        let cluster = self
            .cluster_manager
            .get_cluster_by_id(cluster_id)
            .ok_or_else(|| anyhow!("Cluster not found: {}", cluster_id))?;

        // Source 2: BKS recommended techniques (from our own cache, not library's)
        let bks_techniques = self.get_bks_recommended_techniques(cluster_id).await?;

        // Source 3: PKS user preferences (if available)
        let pks_techniques = self.get_pks_preferred_techniques(cluster_id).await?;

        // Build selection directly from cluster techniques
        let mut selected = Vec::new();

        // 1. Always include role playing (paper's rule)
        if let Some(role) = self.library.get(&PromptingTechnique::RolePlaying)
            && role.min_seal_quality <= seal_quality
        {
            selected.push(role);
        }

        // 2. Select emotional stimulus from cluster techniques
        let emotion_options: Vec<&TechniqueMetadata> = cluster
            .techniques
            .iter()
            .filter_map(|t| self.library.get(t))
            .filter(|t| {
                t.category == TechniqueCategory::EmotionalStimulus
                    && t.min_seal_quality <= seal_quality
            })
            .collect();

        if let Some(emotion) =
            self.select_best_by_priority(&pks_techniques, &bks_techniques, &emotion_options)
        {
            selected.push(emotion);
        }

        // 3. Select reasoning technique
        let reasoning_options: Vec<&TechniqueMetadata> = cluster
            .techniques
            .iter()
            .filter_map(|t| self.library.get(t))
            .filter(|t| {
                t.category == TechniqueCategory::Reasoning && t.min_seal_quality <= seal_quality
            })
            .collect();

        if let Some(reasoning) = self.select_reasoning_by_complexity(
            &pks_techniques,
            &bks_techniques,
            &reasoning_options,
            seal_quality,
        ) {
            selected.push(reasoning);
        }

        // 4. Optionally select "Others" category
        if seal_quality > 0.6 {
            let support_options: Vec<&TechniqueMetadata> = cluster
                .techniques
                .iter()
                .filter_map(|t| self.library.get(t))
                .filter(|t| {
                    t.category == TechniqueCategory::Others && t.min_seal_quality <= seal_quality
                })
                .collect();

            if let Some(support) =
                self.select_best_by_priority(&pks_techniques, &bks_techniques, &support_options)
            {
                selected.push(support);
            }
        }

        Ok(selected)
    }

    /// Select reasoning technique based on complexity
    fn select_reasoning_by_complexity<'a>(
        &self,
        pks: &[PromptingTechnique],
        bks: &[PromptingTechnique],
        options: &[&'a TechniqueMetadata],
        seal_quality: f32,
    ) -> Option<&'a TechniqueMetadata> {
        // Filter by complexity based on SEAL quality
        let complexity = if seal_quality < 0.5 {
            ComplexityLevel::Simple
        } else if seal_quality < 0.8 {
            ComplexityLevel::Moderate
        } else {
            ComplexityLevel::Advanced
        };

        options
            .iter()
            .filter(|t| {
                t.complexity_level == complexity || t.complexity_level == ComplexityLevel::Simple
            })
            .max_by_key(|t| {
                // Prioritize: PKS > BKS > complexity match
                let pks_bonus = if pks.contains(&t.technique) { 100 } else { 0 };
                let bks_bonus = if bks.contains(&t.technique) { 50 } else { 0 };
                let complexity_bonus = if t.complexity_level == complexity {
                    10
                } else {
                    0
                };
                pks_bonus + bks_bonus + complexity_bonus
            })
            .copied()
    }

    /// Get BKS recommended techniques for a cluster
    ///
    /// Queries the BKS cache directly for techniques that have been promoted
    /// based on collective user experience.
    async fn get_bks_recommended_techniques(
        &self,
        cluster_id: &str,
    ) -> Result<Vec<PromptingTechnique>> {
        if let Some(ref bks_cache) = self.bks_cache {
            let bks = bks_cache.lock().await;

            // Query BKS for truths matching the cluster context
            let truths = bks.get_matching_truths(cluster_id);

            let mut recommended = Vec::new();
            for truth in truths {
                // Parse technique from truth rule/rationale
                let text = format!("{} {}", truth.rule, truth.rationale);

                // Check all known techniques (15 from the paper)
                let all_techniques = [
                    PromptingTechnique::RolePlaying,
                    PromptingTechnique::EmotionPrompting,
                    PromptingTechnique::StressPrompting,
                    PromptingTechnique::ChainOfThought,
                    PromptingTechnique::LogicOfThought,
                    PromptingTechnique::LeastToMost,
                    PromptingTechnique::ThreadOfThought,
                    PromptingTechnique::PlanAndSolve,
                    PromptingTechnique::SkeletonOfThought,
                    PromptingTechnique::ScratchpadPrompting,
                    PromptingTechnique::DecomposedPrompting,
                    PromptingTechnique::IgnoreIrrelevantConditions,
                    PromptingTechnique::HighlightedCoT,
                    PromptingTechnique::SkillsInContext,
                    PromptingTechnique::AutomaticInformationFiltering,
                ];

                for technique in all_techniques {
                    if text.contains(technique.to_str()) {
                        recommended.push(technique);
                    }
                }
            }

            Ok(recommended)
        } else {
            Ok(Vec::new())
        }
    }

    /// Get PKS preferred techniques for a cluster
    async fn get_pks_preferred_techniques(
        &self,
        cluster_id: &str,
    ) -> Result<Vec<PromptingTechnique>> {
        if let Some(ref pks_cache) = self.pks_cache {
            let pks = pks_cache.lock().await;

            // Query PKS for user's preferred techniques
            // Example fact key: "preferred_technique:numerical_reasoning"
            let key = format!("preferred_technique:{}", cluster_id);
            if let Some(fact) = pks.get_fact_by_key(&key) {
                // Parse techniques from fact value
                // Example value: "ChainOfThought,PlanAndSolve"
                let techniques: Vec<PromptingTechnique> = fact
                    .value
                    .split(',')
                    .filter_map(|s: &str| PromptingTechnique::parse_id(s.trim()).ok())
                    .collect();
                Ok(techniques)
            } else {
                Ok(Vec::new())
            }
        } else {
            Ok(Vec::new())
        }
    }

    /// Select best technique by priority (PKS > BKS > default)
    fn select_best_by_priority<'a>(
        &self,
        pks: &[PromptingTechnique],
        bks: &[PromptingTechnique],
        options: &[&'a TechniqueMetadata],
    ) -> Option<&'a TechniqueMetadata> {
        options
            .iter()
            .max_by_key(|t| {
                // PKS > BKS > cluster default
                if pks.contains(&t.technique) {
                    2
                } else if bks.contains(&t.technique) {
                    1
                } else {
                    0
                }
            })
            .copied()
    }

    /// Compose prompt from selected techniques
    fn compose_prompt(
        &self,
        task_description: &str,
        techniques: &[&TechniqueMetadata],
        cluster_description: &str,
    ) -> String {
        let mut prompt_parts = Vec::new();

        // Role assignment (always first)
        if let Some(role_technique) = techniques
            .iter()
            .find(|t| t.category == TechniqueCategory::RoleAssignment)
        {
            let role_section = self.apply_technique_template(
                role_technique,
                task_description,
                cluster_description,
            );
            prompt_parts.push(role_section);
        }

        // Emotional stimulus
        if let Some(emotion_technique) = techniques
            .iter()
            .find(|t| t.category == TechniqueCategory::EmotionalStimulus)
        {
            prompt_parts.push(self.apply_technique_template(
                emotion_technique,
                task_description,
                cluster_description,
            ));
        }

        // Reasoning technique
        if let Some(reasoning_technique) = techniques
            .iter()
            .find(|t| t.category == TechniqueCategory::Reasoning)
        {
            prompt_parts.push(self.apply_technique_template(
                reasoning_technique,
                task_description,
                cluster_description,
            ));
        }

        // Additional support techniques
        for technique in techniques
            .iter()
            .filter(|t| t.category == TechniqueCategory::Others)
        {
            prompt_parts.push(self.apply_technique_template(
                technique,
                task_description,
                cluster_description,
            ));
        }

        // Task description
        prompt_parts.push(format!("\n# Task\n\n{}", task_description));

        prompt_parts.join("\n\n")
    }

    /// Apply technique template with variable substitution
    fn apply_technique_template(
        &self,
        technique: &TechniqueMetadata,
        task_description: &str,
        cluster_description: &str,
    ) -> String {
        // Simple template substitution
        let mut result = technique.template.clone();

        // Extract role and domain from cluster/task for Role Playing
        if technique.technique == PromptingTechnique::RolePlaying {
            let (role, domain) = self.infer_role_and_domain(task_description, cluster_description);
            result = result.replace("{role}", &role).replace("{domain}", &domain);
        }

        // Extract task type and quality for Emotion Prompting
        if technique.technique == PromptingTechnique::EmotionPrompting {
            let task_type = self.infer_task_type(task_description);
            let quality = "precision and accuracy";
            result = result
                .replace("{task_type}", &task_type)
                .replace("{quality}", quality);
        }

        result
            .replace("{task}", task_description)
            .replace("{cluster}", cluster_description)
    }

    /// Infer role and domain from task/cluster description
    fn infer_role_and_domain(
        &self,
        task_description: &str,
        cluster_description: &str,
    ) -> (String, String) {
        let task_lower = task_description.to_lowercase();
        let cluster_lower = cluster_description.to_lowercase();

        // Simple heuristics
        if task_lower.contains("code")
            || task_lower.contains("function")
            || task_lower.contains("implement")
        {
            (
                "software engineer".to_string(),
                "software development".to_string(),
            )
        } else if task_lower.contains("algorithm") || task_lower.contains("optimize") {
            (
                "computer scientist".to_string(),
                "algorithms and data structures".to_string(),
            )
        } else if task_lower.contains("calculate") || task_lower.contains("numerical") {
            (
                "mathematician".to_string(),
                "numerical analysis".to_string(),
            )
        } else if task_lower.contains("analyze") || task_lower.contains("understand") {
            ("analyst".to_string(), "problem analysis".to_string())
        } else if cluster_lower.contains("code") {
            ("developer".to_string(), "software engineering".to_string())
        } else {
            ("expert".to_string(), "problem solving".to_string())
        }
    }

    /// Infer task type for Emotion Prompting
    fn infer_task_type(&self, task_description: &str) -> String {
        let task_lower = task_description.to_lowercase();

        if task_lower.contains("calculate") || task_lower.contains("compute") {
            "calculation".to_string()
        } else if task_lower.contains("implement") || task_lower.contains("create") {
            "implementation".to_string()
        } else if task_lower.contains("analyze") || task_lower.contains("understand") {
            "analysis".to_string()
        } else if task_lower.contains("fix") || task_lower.contains("debug") {
            "debugging".to_string()
        } else {
            "task".to_string()
        }
    }

    /// Get reference to cluster manager
    pub fn cluster_manager(&self) -> &TaskClusterManager {
        &self.cluster_manager
    }

    /// Get mutable reference to cluster manager
    pub fn cluster_manager_mut(&mut self) -> &mut TaskClusterManager {
        &mut self.cluster_manager
    }
}

/// Result of prompt generation
#[derive(Debug, Clone)]
pub struct GeneratedPrompt {
    /// The generated system prompt text
    pub system_prompt: String,
    /// ID of the cluster that was matched
    pub cluster_id: String,
    /// Techniques that were selected
    pub techniques: Vec<PromptingTechnique>,
    /// SEAL quality score used
    pub seal_quality: f32,
    /// Similarity score to matched cluster
    pub similarity_score: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clustering::TaskCluster;

    #[test]
    fn test_infer_role_and_domain() {
        let generator = PromptGenerator::new(TechniqueLibrary::new(), TaskClusterManager::new());

        let (role, domain) = generator.infer_role_and_domain("Implement a function", "code");
        assert_eq!(role, "software engineer");
        assert_eq!(domain, "software development");

        let (role, domain) = generator.infer_role_and_domain("Calculate prime numbers", "math");
        assert_eq!(role, "mathematician");
        assert_eq!(domain, "numerical analysis");
    }

    #[test]
    fn test_infer_task_type() {
        let generator = PromptGenerator::new(TechniqueLibrary::new(), TaskClusterManager::new());

        assert_eq!(
            generator.infer_task_type("Calculate the sum"),
            "calculation"
        );
        assert_eq!(
            generator.infer_task_type("Implement a class"),
            "implementation"
        );
        assert_eq!(generator.infer_task_type("Analyze the code"), "analysis");
        assert_eq!(generator.infer_task_type("Fix the bug"), "debugging");
    }

    #[tokio::test]
    async fn test_prompt_generation_basic() {
        let mut cluster_manager = TaskClusterManager::new();

        // Create a test cluster
        let cluster = TaskCluster::new(
            "test_cluster".to_string(),
            "Code generation tasks".to_string(),
            vec![0.5; 768],
            vec![
                PromptingTechnique::RolePlaying,
                PromptingTechnique::EmotionPrompting,
                PromptingTechnique::ChainOfThought,
            ],
            vec!["Write a function".to_string()],
        );
        cluster_manager.add_cluster(cluster);

        let generator = PromptGenerator::new(TechniqueLibrary::new(), cluster_manager);

        // Generate prompt
        let task_embedding = vec![0.5; 768];
        let result = generator
            .generate_prompt("Write a function to sort an array", &task_embedding, None)
            .await
            .unwrap();

        // Verify structure
        assert!(!result.system_prompt.is_empty());
        assert_eq!(result.cluster_id, "test_cluster");
        assert!(!result.techniques.is_empty());
    }
}
