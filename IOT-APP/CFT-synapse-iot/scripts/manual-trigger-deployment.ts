/**
 * Manual Deployment Trigger Script
 * 
 * This script reads pending deployments from Supabase and writes them to DynamoDB
 * to trigger the AutoDeployModelToPi Lambda function.
 * 
 * Use this as a workaround when CORS prevents frontend from writing to DynamoDB.
 * 
 * Usage:
 *   npx ts-node scripts/manual-trigger-deployment.ts
 */

import { createClient } from '@supabase/supabase-js';

const SUPABASE_URL = process.env.VITE_SUPABASE_URL || 'https://zpmbvtfptddmbxhmzapz.supabase.co';
const SUPABASE_ANON_KEY = process.env.VITE_SUPABASE_ANON_KEY || 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InpwbWJ2dGZwdGRkbWJ4aG16YXB6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzQxMDE4MTYsImV4cCI6MjA4OTY3NzgxNn0.jdmdDLKf2xEkb9pJIl-Mc3MgJD_BttQieiknNhr6cT8';
const API_GATEWAY_URL = process.env.VITE_API_GATEWAY_URL || 'https://fbujw415e6.execute-api.ap-southeast-2.amazonaws.com/prod';

const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

interface DeploymentRow {
  deployment_id: string;
  deployment_model_version_id: string;
  deployment_status: string;
  target_node_id: string;
  deployment_created_at: string;
  deployment_requested_by: string;
}

async function triggerPendingDeployments() {
  console.log('[Manual Trigger] Fetching pending deployments from Supabase...');
  
  // Get all pending deployments
  const { data: deployments, error } = await supabase
    .from('deployments_all')
    .select('*')
    .eq('deployment_status', 'pending')
    .order('deployment_created_at', { ascending: true });
  
  if (error) {
    console.error('[Manual Trigger] Error fetching deployments:', error);
    return;
  }
  
  if (!deployments || deployments.length === 0) {
    console.log('[Manual Trigger] No pending deployments found');
    return;
  }
  
  console.log(`[Manual Trigger] Found ${deployments.length} pending deployment(s)`);
  
  // Write to DynamoDB via API Gateway
  for (const deployment of deployments as DeploymentRow[]) {
    console.log(`[Manual Trigger] Triggering deployment ${deployment.deployment_id} for node ${deployment.target_node_id}...`);
    
    try {
      const response = await fetch(`${API_GATEWAY_URL}/dynamodb/deployments`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          deployment_id: deployment.deployment_id,
          deployment_requested_by: deployment.deployment_requested_by,
          deployment_model_version_id: deployment.deployment_model_version_id,
          deployment_status: deployment.deployment_status,
          target_node_id: deployment.target_node_id,
          deployment_created_at: deployment.deployment_created_at,
        }),
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        console.error(`[Manual Trigger] Failed to write deployment ${deployment.deployment_id}:`, errorText);
        continue;
      }
      
      const result = await response.json();
      console.log(`[Manual Trigger] ✅ Successfully triggered deployment ${deployment.deployment_id}`);
      console.log(`[Manual Trigger] Response:`, result);
      
    } catch (err) {
      console.error(`[Manual Trigger] Error triggering deployment ${deployment.deployment_id}:`, err);
    }
  }
  
  console.log('[Manual Trigger] Done!');
}

// Run the script
triggerPendingDeployments().catch(console.error);
