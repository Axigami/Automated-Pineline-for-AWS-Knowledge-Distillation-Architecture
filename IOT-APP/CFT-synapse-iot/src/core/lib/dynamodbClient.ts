/**
 * DynamoDB Client
 * Provides methods to write to DynamoDB tables via API Gateway
 * 
 * Architecture:
 * Frontend → API Gateway → Lambda → DynamoDB
 * 
 * This ensures data is written to BOTH Supabase (for frontend reads) 
 * and DynamoDB (for Lambda processing via Streams)
 */

const API_BASE_URL = import.meta.env.VITE_API_GATEWAY_URL || 'https://fbujw415e6.execute-api.ap-southeast-2.amazonaws.com/prod';

interface DynamoDBRetrainJob {
  job_id: string;
  job_requested_by?: string;
  job_home_id?: string;
  job_status: 'queued' | 'running' | 'completed' | 'failed';
  job_data_range?: string;
  job_epochs?: number;
  job_knowledge_distillation?: boolean;
  job_progress_percent?: number;
  job_created_at: string;
  job_pipeline_steps_json?: string;
  job_training_batch_size?: number;
  job_training_learning_rate?: number;
}

interface DynamoDBDeployment {
  deployment_id: string;
  deployment_requested_by?: string;
  deployment_model_version_id: string;
  deployment_status: 'pending' | 'in_progress' | 'deployed' | 'failed';
  deployment_created_at: string;
  target_node_id: string;
}

interface WriteRetrainJobResponse {
  success: boolean;
  job_id: string;
  message?: string;
}

interface WriteDeploymentResponse {
  success: boolean;
  deployment_ids: string[];
  message?: string;
}

class DynamoDBClient {
  private baseUrl: string;
  private authToken: string | null = null;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  /**
   * Set authentication token for API requests
   */
  setAuthToken(token: string) {
    this.authToken = token;
  }

  /**
   * Get authorization headers
   */
  private getHeaders(): HeadersInit {
    const headers: HeadersInit = {
      'Content-Type': 'application/json',
    };

    if (this.authToken) {
      headers['Authorization'] = `Bearer ${this.authToken}`;
    }

    return headers;
  }

  /**
   * Write retrain job to DynamoDB
   * This triggers the RetrainJobHandler Lambda via DynamoDB Stream
   */
  async writeRetrainJob(job: DynamoDBRetrainJob): Promise<WriteRetrainJobResponse> {
    try {
      const response = await fetch(`${this.baseUrl}/dynamodb/retrain-jobs`, {
        method: 'POST',
        headers: this.getHeaders(),
        body: JSON.stringify(job),
      });

      if (!response.ok) {
        const error = await response.json().catch(() => ({ error: 'Unknown error' }));
        throw new Error(error.error || 'Failed to write retrain job to DynamoDB');
      }

      return response.json();
    } catch (error) {
      console.error('Failed to write to DynamoDB:', error);
      // Return success=false but don't throw - allow Supabase write to succeed
      return {
        success: false,
        job_id: job.job_id,
        message: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  }

  /**
   * Write deployment to DynamoDB
   * This triggers the AutoDeployModelToPi Lambda via DynamoDB Stream
   */
  async writeDeployment(deployment: DynamoDBDeployment): Promise<WriteDeploymentResponse> {
    try {
      const response = await fetch(`${this.baseUrl}/dynamodb/deployments`, {
        method: 'POST',
        headers: this.getHeaders(),
        body: JSON.stringify(deployment),
      });

      if (!response.ok) {
        const error = await response.json().catch(() => ({ error: 'Unknown error' }));
        throw new Error(error.error || 'Failed to write deployment to DynamoDB');
      }

      return response.json();
    } catch (error) {
      console.error('Failed to write to DynamoDB:', error);
      // Return success=false but don't throw - allow Supabase write to succeed
      return {
        success: false,
        deployment_ids: [deployment.deployment_id],
        message: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  }

  /**
   * Write multiple deployments to DynamoDB (batch)
   */
  async writeDeployments(deployments: DynamoDBDeployment[]): Promise<WriteDeploymentResponse> {
    try {
      const response = await fetch(`${this.baseUrl}/dynamodb/deployments/batch`, {
        method: 'POST',
        headers: this.getHeaders(),
        body: JSON.stringify({ deployments }),
      });

      if (!response.ok) {
        const error = await response.json().catch(() => ({ error: 'Unknown error' }));
        throw new Error(error.error || 'Failed to write deployments to DynamoDB');
      }

      return response.json();
    } catch (error) {
      console.error('Failed to write to DynamoDB:', error);
      // Return success=false but don't throw - allow Supabase write to succeed
      return {
        success: false,
        deployment_ids: deployments.map(d => d.deployment_id),
        message: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  }
}

// Export singleton instance
export const dynamodbClient = new DynamoDBClient();

// Export types
export type {
  DynamoDBRetrainJob,
  DynamoDBDeployment,
  WriteRetrainJobResponse,
  WriteDeploymentResponse,
};
