// Core Layer – Shared infrastructure exports

// Supabase singleton & types
export { supabase } from './lib/supabaseClient';
export type { Database } from './lib/database.types';

// Models
export * from './models/auth.types';
export * from './models/api.types';

// Views (Atoms/Molecules)
export { BaseLayout } from './views/BaseLayout';
export { LoadingSpinner } from './views/LoadingSpinner';

// Controllers (Global Hooks)
export { useSupabase } from './controllers/useApiClient';
export { useWebSocket } from './controllers/useWebSocket';
