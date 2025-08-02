// TypeScript type definitions for BeatWizard

export interface User {
  id: string;
  email: string;
  full_name?: string;
  avatar_url?: string;
  subscription_tier: 'free' | 'pro' | 'premium';
  created_at: string;
  updated_at: string;
}

export interface AnalysisResult {
  id: string;
  user_id: string;
  file_name: string;
  file_url?: string;
  analysis_data: AnalysisData;
  skill_level: 'beginner' | 'intermediate' | 'advanced';
  genre?: string;
  created_at: string;
}

export interface AnalysisData {
  tempo_analysis: {
    primary_tempo: number;
    confidence: number;
    tempo_category: string;
    secondary_tempos?: number[];
  };
  key_analysis: {
    primary_key: string;
    confidence: number;
    key_profile?: number[];
    modality?: string;
  };
  frequency_analysis: {
    bands: {
      sub_bass: FrequencyBand;
      bass: FrequencyBand;
      low_mid: FrequencyBand;
      mid: FrequencyBand;
      high_mid: FrequencyBand;
      presence: FrequencyBand;
      brilliance: FrequencyBand;
    };
    overall_assessment: {
      quality_rating: string;
      balance_score: number;
      issues: string[];
      recommendations: string[];
    };
  };
  loudness_analysis: {
    integrated_loudness: number;
    loudness_range: number;
    dynamic_range_analysis: {
      dynamic_range_quality: string;
      peak_to_rms_ratio: number;
    };
    streaming_compliance: {
      spotify_compliant: boolean;
      apple_music_compliant: boolean;
      youtube_compliant: boolean;
      target_adjustments: Record<string, number>;
    };
  };
  stereo_analysis: {
    overall_assessment: {
      overall_quality: string;
      stereo_width: number;
      phase_correlation: number;
    };
    mid_side_analysis: {
      mid_energy: number;
      side_energy: number;
      width_score: number;
    };
  };
  sound_selection_analysis?: {
    kick_analysis: ElementAnalysis;
    snare_analysis: ElementAnalysis;
    bass_analysis: ElementAnalysis;
    melody_analysis: ElementAnalysis;
    overall_sound_selection: {
      overall_quality: string;
      quality_score: number;
      feedback: string;
    };
    recommendations: {
      priority_recommendations: PriorityRecommendation[];
      splice_pack_suggestions: string[];
      beginner_tips: string[];
      next_steps: string[];
    };
  };
  rhythm_analysis?: {
    beat_grid: {
      quality: string;
      beat_consistency: number;
      issues: string[];
    };
    groove: {
      groove_type: string;
      groove_strength: number;
      swing: {
        swing_ratio: number;
        swing_type: string;
      };
    };
    overall_rhythm: {
      overall_quality: string;
      rhythm_score: number;
      strengths: string[];
      weaknesses: string[];
    };
  };
  harmony_analysis?: {
    chord_analysis: {
      detected_chords: string[];
      unique_chords: string[];
      chord_changes: number;
      chord_quality: string;
      issues: string[];
    };
    progression_analysis: {
      common_progressions: any[];
      progression_quality: string;
    };
    overall_harmony: {
      overall_quality: string;
      harmony_score: number;
      strengths: string[];
      weaknesses: string[];
    };
  };
  intelligent_feedback?: IntelligentFeedback;
  overall_assessment: {
    overall_quality: string;
    technical_quality: string;
    mix_quality: string;
    mastering_readiness: boolean;
    commercial_readiness: boolean;
    streaming_readiness: boolean;
    quality_score: number;
  };
  analysis_metadata: {
    file_name: string;
    file_duration: number;
    sample_rate: number;
    channels: number;
    analysis_duration: number;
    analysis_timestamp: number;
    analysis_version: string;
  };
}

export interface FrequencyBand {
  frequency_range: [number, number];
  energy_level: number;
  quality_rating: string;
  issues: string[];
  recommendations: string[];
}

export interface ElementAnalysis {
  quality: string;
  issues: string[];
  recommendations: string[];
  splice_suggestions: string[];
  level_analysis?: {
    rms: number;
    peak?: number;
  };
  frequency_distribution?: Record<string, number>;
}

export interface PriorityRecommendation {
  element: string;
  priority: 'high' | 'medium' | 'low';
  issue: string;
  solution: string;
  beginner_tip?: string;
}

export interface IntelligentFeedback {
  production_feedback: {
    priority_issues: PriorityRecommendation[];
    quick_wins: QuickWin[];
    detailed_feedback: string[];
    encouragement: string[];
  };
  sound_selection: {
    element_feedback: Record<string, any>;
    sample_recommendations: Record<string, string[]>;
    layering_suggestions: string[];
    splice_pack_suggestions: string[];
  };
  arrangement: {
    structure_analysis: Record<string, any>;
    arrangement_suggestions: string[];
    energy_curve: Record<string, any>;
    transition_ideas: string[];
  };
  technical: {
    mixing_priority: any[];
    mastering_readiness: Record<string, any>;
    streaming_compliance: Record<string, any>;
    technical_improvements: string[];
  };
  creative_suggestions: {
    sound_design: string[];
    arrangement_ideas: string[];
    genre_specific: string[];
    experimental: string[];
  };
  learning_path: {
    immediate_focus: string[];
    next_month: string[];
    long_term_goals: string[];
    recommended_resources: string[];
  };
  references: Reference[];
  sample_suggestions: Record<string, string[]>;
  overall_assessment: {
    overall_rating: string;
    strengths: string[];
    main_weaknesses: string[];
    commercial_potential: string;
    next_version_focus: string[];
    motivational_message: string;
  };
}

export interface QuickWin {
  action: string;
  description: string;
  steps: string[];
}

export interface Reference {
  artist: string;
  track: string;
  why: string;
  focus_areas: string[];
}

export interface UploadProgress {
  progress: number;
  stage: 'uploading' | 'processing' | 'analyzing' | 'generating_feedback' | 'complete';
  message: string;
}

export interface AnalysisSettings {
  skill_level: 'beginner' | 'intermediate' | 'advanced';
  genre: string;
  include_advanced_features: boolean;
  focus_areas: string[];
}

export interface DashboardStats {
  total_analyses: number;
  avg_quality_score: number;
  improvement_trend: number;
  recent_analyses: AnalysisResult[];
  skill_progression: {
    mixing_score: number;
    arrangement_score: number;
    sound_selection_score: number;
    technical_score: number;
  };
}

export type Theme = 'light' | 'dark' | 'system';

export interface AppConfig {
  theme: Theme;
  beginner_mode: boolean;
  auto_analysis: boolean;
  notification_preferences: {
    analysis_complete: boolean;
    weekly_progress: boolean;
    tips_and_tutorials: boolean;
  };
}