-- BeatWizard Database Schema
-- Run this in your Supabase SQL Editor

-- Users table (extends Supabase auth.users)
CREATE TABLE public.user_profiles (
  id UUID REFERENCES auth.users(id) PRIMARY KEY,
  username TEXT UNIQUE,
  skill_level TEXT CHECK (skill_level IN ('beginner', 'intermediate', 'advanced')) DEFAULT 'beginner',
  preferred_genres TEXT[],
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Audio analysis results table
CREATE TABLE public.audio_analyses (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID REFERENCES auth.users(id),
  file_name TEXT NOT NULL,
  file_url TEXT NOT NULL,
  
  -- Basic analysis
  tempo_bpm REAL,
  tempo_confidence REAL,
  key_signature TEXT,
  key_confidence REAL,
  
  -- Professional analysis
  lufs REAL,
  lra REAL,
  peak_to_rms REAL,
  stereo_width REAL,
  phase_correlation REAL,
  
  -- Frequency analysis (7 bands)
  sub_bass REAL,
  bass REAL,
  low_mids REAL,
  mids REAL,
  high_mids REAL,
  presence REAL,
  brilliance REAL,
  
  -- AI feedback
  ai_feedback JSONB,
  skill_level_feedback JSONB,
  genre_specific_feedback JSONB,
  
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- File storage policies
-- Enable RLS
ALTER TABLE public.user_profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.audio_analyses ENABLE ROW LEVEL SECURITY;

-- User profiles policies
CREATE POLICY "Users can view own profile" ON public.user_profiles
  FOR SELECT USING (auth.uid() = id);

CREATE POLICY "Users can update own profile" ON public.user_profiles
  FOR UPDATE USING (auth.uid() = id);

CREATE POLICY "Users can insert own profile" ON public.user_profiles
  FOR INSERT WITH CHECK (auth.uid() = id);

-- Audio analyses policies
CREATE POLICY "Users can view own analyses" ON public.audio_analyses
  FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own analyses" ON public.audio_analyses
  FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own analyses" ON public.audio_analyses
  FOR UPDATE USING (auth.uid() = user_id);

-- Storage bucket policies (run in Supabase Storage)
-- Create bucket: audio-files
-- Policy: "Users can upload their own audio files"
-- Policy: "Users can view their own audio files"