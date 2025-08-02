# BeatWizard React Frontend Setup Guide

## 🚀 Complete Setup Instructions

### 1. Create React TypeScript Project

```bash
npx create-react-app beatwizard-frontend --template typescript
cd beatwizard-frontend
```

### 2. Install Dependencies

```bash
# Core dependencies
npm install @supabase/supabase-js
npm install @types/file-saver file-saver
npm install react-router-dom @types/react-router-dom
npm install react-hook-form @hookform/resolvers yup
npm install react-hot-toast
npm install lucide-react
npm install clsx tailwind-merge

# Tailwind CSS
npm install -D tailwindcss postcss autoprefixer @tailwindcss/forms
npx tailwindcss init -p
```

### 3. Supabase Database Schema

Run these SQL commands in your Supabase SQL editor:

```sql
-- Enable RLS
alter table if exists public.profiles enable row level security;
alter table if exists public.analysis_results enable row level security;

-- Create profiles table
create table public.profiles (
  id uuid references auth.users on delete cascade,
  email text,
  full_name text,
  avatar_url text,
  subscription_tier text default 'free',
  created_at timestamp with time zone default timezone('utc'::text, now()) not null,
  updated_at timestamp with time zone default timezone('utc'::text, now()) not null,
  primary key (id)
);

-- Create analysis_results table
create table public.analysis_results (
  id uuid default uuid_generate_v4() primary key,
  user_id uuid references public.profiles(id) on delete cascade,
  file_name text not null,
  file_url text,
  analysis_data jsonb not null,
  skill_level text default 'beginner',
  genre text,
  created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- Create storage bucket for audio files
insert into storage.buckets (id, name, public) values ('audio-files', 'audio-files', false);

-- Set up Row Level Security policies
create policy "Users can view own profile" on profiles for select using (auth.uid() = id);
create policy "Users can update own profile" on profiles for update using (auth.uid() = id);
create policy "Users can insert own profile" on profiles for insert with check (auth.uid() = id);

create policy "Users can view own analysis results" on analysis_results for select using (auth.uid() = user_id);
create policy "Users can insert own analysis results" on analysis_results for insert with check (auth.uid() = user_id);

-- Storage policies
create policy "Users can upload own audio files" on storage.objects for insert with check (bucket_id = 'audio-files' AND auth.uid()::text = (storage.foldername(name))[1]);
create policy "Users can view own audio files" on storage.objects for select using (bucket_id = 'audio-files' AND auth.uid()::text = (storage.foldername(name))[1]);
create policy "Users can delete own audio files" on storage.objects for delete using (bucket_id = 'audio-files' AND auth.uid()::text = (storage.foldername(name))[1]);

-- Function to handle new user registration
create or replace function public.handle_new_user() 
returns trigger as $$
begin
  insert into public.profiles (id, email, full_name)
  values (new.id, new.email, new.raw_user_meta_data->>'full_name');
  return new;
end;
$$ language plpgsql security definer;

-- Trigger for new user registration
create trigger on_auth_user_created
  after insert on auth.users
  for each row execute procedure public.handle_new_user();
```

### 4. Environment Variables

Create `.env.local`:

```env
REACT_APP_SUPABASE_URL=https://rrmoicsnssfbflkbqcek.supabase.co
REACT_APP_SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJybW9pY3Nuc3NmYmZsa2JxY2VrIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTM5MzU2NjgsImV4cCI6MjA2OTUxMTY2OH0.hQvp5-KQ3NunGvt7rayrXLAgr7GG4O49brkNVDhJO88
REACT_APP_BEATWIZARD_API_URL=http://localhost:8080
```

### 5. Project Structure

```
src/
├── components/
│   ├── ui/                 # Reusable UI components
│   ├── auth/              # Authentication components  
│   ├── upload/            # File upload components
│   ├── analysis/          # Analysis display components
│   ├── dashboard/         # Dashboard components
│   └── layout/            # Layout components
├── hooks/                 # Custom React hooks
├── lib/                   # Utilities and configurations
├── pages/                 # Page components
├── types/                 # TypeScript type definitions
└── utils/                 # Helper functions
```

This setup provides:
- ✅ Complete Supabase integration
- ✅ User authentication system
- ✅ File upload to Supabase storage
- ✅ Database storage for analysis results
- ✅ Row-level security
- ✅ TypeScript support
- ✅ Tailwind CSS styling
- ✅ Mobile-responsive design
- ✅ Professional project structure