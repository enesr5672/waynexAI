-- WAYNEX AI Production Database Setup
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_predictions_batch_id ON predictions(batch_id);
CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at);
CREATE INDEX IF NOT EXISTS idx_realtime_data_sensor_id ON realtime_data(sensor_id);
CREATE INDEX IF NOT EXISTS idx_realtime_data_created_at ON realtime_data(created_at);
CREATE INDEX IF NOT EXISTS idx_system_metrics_recorded_at ON system_metrics(recorded_at);

-- Create initial admin user (in production, use proper hashing)
INSERT INTO users (username, password_hash, role) 
VALUES ('waynex_admin', 'WaynexAI2024!', 'admin') 
ON CONFLICT (username) DO NOTHING;
