-- Fix admin role for user dced8a9f-5f89-4d44-a146-9f7070793749
-- This script ensures the user has the 'admin' role in users_roles_settings table

-- First, check if the user exists in users_roles_settings
-- If not, insert a new row with admin role
-- If yes, update the role_code to 'admin'

-- Option 1: Insert if not exists (PostgreSQL UPSERT)
INSERT INTO users_roles_settings (
  user_id,
  role_code,
  user_created_at
)
VALUES (
  'dced8a9f-5f89-4d44-a146-9f7070793749',
  'admin',
  NOW()
)
ON CONFLICT (user_id) 
DO UPDATE SET 
  role_code = 'admin';

-- Verify the update
SELECT 
  user_id,
  user_email,
  role_code,
  role_name
FROM users_roles_settings
WHERE user_id = 'dced8a9f-5f89-4d44-a146-9f7070793749';
