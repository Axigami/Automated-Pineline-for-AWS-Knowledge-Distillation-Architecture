import { useEffect, useState } from 'react';
import { supabase } from '../../../core/lib/supabaseClient';

export interface RetrainingJob {
  job_id: string;
  job_status: string;
  job_progress_percent: number;
  job_epochs: number;
  job_data_range: string;
  job_knowledge_distillation: boolean;
  job_training_batch_size: number;
  job_training_learning_rate: number;
  job_started_at: string | null;
  job_finished_at: string | null;
  event_step: string | null;
  event_message: string | null;
  teacher_version: string;
  student_version: string;
}

export interface RetrainingData {
  job: RetrainingJob | null;
  loading: boolean;
  error: string | null;
}

export function useRetrainingData(): RetrainingData {
  const [data, setData] = useState<RetrainingData>({ job: null, loading: true, error: null });

  useEffect(() => {
    let cancelled = false;
    async function fetch() {
      try {
        const { data: job, error } = await supabase
          .from('retrain_jobs_all')
          .select(`
            job_id, job_status, job_progress_percent, job_epochs,
            job_data_range, job_knowledge_distillation,
            job_training_batch_size, job_training_learning_rate,
            job_started_at, job_finished_at,
            event_step, event_message,
            job_teacher_from_version_id, job_student_to_version_id
          `)
          .order('job_created_at', { ascending: false })
          .limit(1)
          .maybeSingle();

        if (error) throw new Error(error.message);
        if (cancelled) return;

        if (!job) {
          setData({ job: null, loading: false, error: null });
          return;
        }

        // Lookup teacher version label
        let teacherVersion = '—';
        if (job.job_teacher_from_version_id) {
          const { data: tv } = await supabase
            .from('model_versions')
            .select('version')
            .eq('id', job.job_teacher_from_version_id)
            .maybeSingle();
          teacherVersion = tv?.version ?? '—';
        }

        // Lookup student version label
        let studentVersion = '—';
        if (job.job_student_to_version_id) {
          const { data: sv } = await supabase
            .from('model_versions')
            .select('version')
            .eq('id', job.job_student_to_version_id)
            .maybeSingle();
          studentVersion = sv?.version ?? '—';
        }

        const result: RetrainingJob = {
          job_id: job.job_id,
          job_status: job.job_status ?? 'pending',
          job_progress_percent: job.job_progress_percent ?? 0,
          job_epochs: job.job_epochs ?? 50,
          job_data_range: job.job_data_range ?? '30d',
          job_knowledge_distillation: job.job_knowledge_distillation ?? false,
          job_training_batch_size: job.job_training_batch_size ?? 128,
          job_training_learning_rate: job.job_training_learning_rate ?? 0.001,
          job_started_at: job.job_started_at,
          job_finished_at: job.job_finished_at,
          event_step: job.event_step,
          event_message: job.event_message,
          teacher_version: teacherVersion,
          student_version: studentVersion,
        };

        if (!cancelled) setData({ job: result, loading: false, error: null });
      } catch (e: unknown) {
        if (!cancelled) {
          const msg = e instanceof Error ? e.message : String(e);
          setData({ job: null, loading: false, error: msg });
        }
      }
    }
    fetch();
    return () => { cancelled = true; };
  }, []);

  return data;
}
