/**
 * Lambda API Client
 * Provides methods to invoke Lambda functions via API Gateway
 */

const API_BASE_URL = import.meta.env.VITE_API_GATEWAY_URL || 'https://fbujw415e6.execute-api.ap-southeast-2.amazonaws.com/prod';

interface TriggerFineTuningParams {
  triggered_by: 'model_insights_ui' | 'mlops_ui';
  sample_count?: number;
  hyperparameters?: {
    batch_size?: number;
    learning_rate?: number;
  };
  home_id?: string;
  user_id?: string;
}

interface TriggerFineTuningResponse {
  message: string;
  job_id: string;
  job_name: string;
  training_data: string;
  timestamp: string;
  triggered_by: string;
  sample_count: number;
}

interface TrainingQueueStatus {
  pending_count: number;
  relabeled_count: number;
  used_count: number;
  total_count: number;
  by_label: Record<string, number>;
  ready_for_training: boolean;
}

interface AddToQueueFlow {
  flow_id: string;
  predicted_label: string;
  corrected_label: string;
  raw_flow: any;
  home_id?: string;
  user_id?: string;
}

interface AddToQueueResponse {
  message: string;
  added_count: number;
  total_count: number;
  errors?: string[];
}

/** Gửi tới API Gateway khi backend có route POST /verify-alert (Lambda CNN-LSTM). */
export interface VerifyCloudAlertParams {
  alert_id: string;
  home_id?: string | null;
}

class LambdaClient {
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
   * Trigger fine-tuning job
   */
  async triggerFineTuning(params: TriggerFineTuningParams): Promise<TriggerFineTuningResponse> {
    const response = await fetch(`${this.baseUrl}/trigger-finetuning`, {
      method: 'POST',
      headers: this.getHeaders(),
      body: JSON.stringify(params),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Failed to trigger fine-tuning');
    }

    return response.json();
  }

  /**
   * Query training queue status
   */
  async getTrainingQueueStatus(): Promise<TrainingQueueStatus> {
    const response = await fetch(`${this.baseUrl}/training-queue`, {
      method: 'GET',
      headers: this.getHeaders(),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Failed to query training queue');
    }

    return response.json();
  }

  /**
   * Add flows to training queue
   */
  async addToTrainingQueue(flows: AddToQueueFlow[]): Promise<AddToQueueResponse> {
    const response = await fetch(`${this.baseUrl}/add-to-queue`, {
      method: 'POST',
      headers: this.getHeaders(),
      body: JSON.stringify({ flows }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Failed to add flows to training queue');
    }

    return response.json();
  }

  /**
   * Tùy chọn: xác minh alert qua Lambda (CNN-LSTM). Nếu API chưa deploy (404) → trả về false, UI vẫn ghi Supabase.
   */
  async verifyCloudAlert(params: VerifyCloudAlertParams): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/verify-alert`, {
        method: 'POST',
        headers: this.getHeaders(),
        body: JSON.stringify(params),
      });
      if (response.status === 404) return false;
      if (!response.ok) return false;
      return true;
    } catch {
      return false;
    }
  }
}

// Export singleton instance
export const lambdaClient = new LambdaClient();

// Export types
export type {
  TriggerFineTuningParams,
  TriggerFineTuningResponse,
  TrainingQueueStatus,
  AddToQueueFlow,
  AddToQueueResponse,
};
