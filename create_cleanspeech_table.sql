-- SQL to create the CleanSpeechDataset table
-- Run this in your PostgreSQL database

-- Create the Recording table first (if it doesn't exist)
-- Note: This table should be created by Django migrations, but here's the structure for reference
CREATE TABLE IF NOT EXISTS data_recording (
    id BIGSERIAL PRIMARY KEY,
    recording_type VARCHAR(50) NOT NULL DEFAULT 'clean_speech' CHECK (recording_type IN ('clean_speech', 'english_language', 'scripted_speech')),
    status VARCHAR(50) NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
    device_info JSONB NULL,
    contributor_id BIGINT NULL REFERENCES authentication_customuser(id) DEFERRABLE INITIALLY DEFERRED,
    approved BOOLEAN NOT NULL DEFAULT FALSE,
    approved_by_id BIGINT NULL REFERENCES authentication_customuser(id) DEFERRABLE INITIALLY DEFERRED,
    approved_at TIMESTAMP NULL,
    audio VARCHAR(100) NULL,
    duration DOUBLE PRECISION NULL,
    recording_date TIMESTAMP NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for Recording table
CREATE INDEX IF NOT EXISTS idx_recording_contributor ON data_recording(contributor_id);
CREATE INDEX IF NOT EXISTS idx_recording_approved_by ON data_recording(approved_by_id);
CREATE INDEX IF NOT EXISTS idx_recording_recording_type ON data_recording(recording_type);
CREATE INDEX IF NOT EXISTS idx_recording_status ON data_recording(status);

-- Create the CleanSpeechDataset table
CREATE TABLE IF NOT EXISTS data_cleanspeechdataset (
    id BIGSERIAL PRIMARY KEY,
    name VARCHAR(255) NULL,
    description TEXT NULL,
    recording_date TIMESTAMP NULL,
    recording_device VARCHAR(255) NOT NULL,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    clean_speech_id VARCHAR(255) NULL UNIQUE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    audio VARCHAR(100) NULL,
    category_id BIGINT NULL REFERENCES core_cleanspeechcategory(id) DEFERRABLE INITIALLY DEFERRED,
    class_name_id BIGINT NULL REFERENCES core_cleanspeechclass(id) DEFERRABLE INITIALLY DEFERRED,
    collector_id BIGINT NULL REFERENCES authentication_customuser(id) DEFERRABLE INITIALLY DEFERRED,
    community_id BIGINT NULL REFERENCES core_community(id) DEFERRABLE INITIALLY DEFERRED,
    dataset_type_id BIGINT NULL REFERENCES data_dataset(id) DEFERRABLE INITIALLY DEFERRED,
    microphone_type_id BIGINT NULL REFERENCES core_microphone_type(id) DEFERRABLE INITIALLY DEFERRED,
    recording_id BIGINT NULL REFERENCES data_recording(id) DEFERRABLE INITIALLY DEFERRED,
    region_id BIGINT NULL REFERENCES core_region(id) DEFERRABLE INITIALLY DEFERRED,
    subclass_id BIGINT NULL REFERENCES core_cleanspeechsubclass(id) DEFERRABLE INITIALLY DEFERRED,
    time_of_day_id BIGINT NULL REFERENCES core_time_of_day(id) DEFERRABLE INITIALLY DEFERRED
);

-- Create indexes for Recording table
CREATE INDEX IF NOT EXISTS idx_recording_contributor ON data_recording(contributor_id);
CREATE INDEX IF NOT EXISTS idx_recording_approved_by ON data_recording(approved_by_id);
CREATE INDEX IF NOT EXISTS idx_recording_recording_type ON data_recording(recording_type);
CREATE INDEX IF NOT EXISTS idx_recording_status ON data_recording(status);

-- Create indexes for CleanSpeechDataset foreign keys
CREATE INDEX IF NOT EXISTS idx_cleanspeech_category ON data_cleanspeechdataset(category_id);
CREATE INDEX IF NOT EXISTS idx_cleanspeech_class ON data_cleanspeechdataset(class_name_id);
CREATE INDEX IF NOT EXISTS idx_cleanspeech_collector ON data_cleanspeechdataset(collector_id);
CREATE INDEX IF NOT EXISTS idx_cleanspeech_community ON data_cleanspeechdataset(community_id);
CREATE INDEX IF NOT EXISTS idx_cleanspeech_dataset_type ON data_cleanspeechdataset(dataset_type_id);
CREATE INDEX IF NOT EXISTS idx_cleanspeech_microphone ON data_cleanspeechdataset(microphone_type_id);
CREATE INDEX IF NOT EXISTS idx_cleanspeech_recording ON data_cleanspeechdataset(recording_id);
CREATE INDEX IF NOT EXISTS idx_cleanspeech_region ON data_cleanspeechdataset(region_id);
CREATE INDEX IF NOT EXISTS idx_cleanspeech_subclass ON data_cleanspeechdataset(subclass_id);
CREATE INDEX IF NOT EXISTS idx_cleanspeech_time_of_day ON data_cleanspeechdataset(time_of_day_id);