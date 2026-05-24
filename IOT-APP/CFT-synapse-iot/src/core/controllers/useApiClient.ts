import { supabase } from '../lib/supabaseClient';

/**
 * useSupabase – Global hook trả về Supabase client singleton.
 * Dùng trong controllers để query DB. Thay thế useApiClient.
 */
export function useSupabase() {
  return supabase;
}
