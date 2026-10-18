"use client";

import { FormEvent, useMemo, useState } from "react";
import type { ClarifyingQuestion } from "@/lib/types";

export interface ClarificationExchange {
  question: string;
  answer: string;
}

interface ClarificationChatProps {
  questions: ClarifyingQuestion[];
  history: ClarificationExchange[];
  busy: boolean;
  onSubmit: (answers: Record<string, string>) => Promise<void>;
}

export function ClarificationChat({ questions, history, busy, onSubmit }: ClarificationChatProps) {
  const initialAnswers = useMemo(
    () => Object.fromEntries(questions.map((item) => [item.question_id, ""])),
    [questions],
  );
  const [answers, setAnswers] = useState<Record<string, string>>(initialAnswers);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (questions.some((question) => !answers[question.question_id]?.trim())) return;
    await onSubmit(Object.fromEntries(
      questions.map((question) => [question.question_id, answers[question.question_id].trim()]),
    ));
    setAnswers(Object.fromEntries(questions.map((question) => [question.question_id, ""])));
  }

  return (
    <section className="mx-auto max-w-3xl space-y-5">
      <header>
        <p className="text-xs font-semibold uppercase tracking-[0.12em] text-indigo-700">Requirement Agent</p>
        <h1 className="mt-2 text-2xl font-semibold tracking-tight text-slate-900">Clarify the project brief</h1>
        <p className="mt-2 text-sm leading-6 text-slate-600">
          Answer the open questions below. Your answers are sent back to the Requirement Agent before planning starts.
        </p>
      </header>

      <div className="space-y-4 rounded-xl border border-slate-200 bg-white p-5 shadow-sm" aria-live="polite">
        <div className="flex justify-end">
          <div className="max-w-[90%] rounded-xl rounded-br-sm bg-indigo-700 px-4 py-3 text-sm leading-6 text-white">
            I submitted the brief and rubric. Please identify anything that needs clarification.
          </div>
        </div>
        {history.map((exchange, index) => (
          <div key={`${index}-${exchange.question}`} className="space-y-3">
            <div className="flex justify-start">
              <div className="max-w-[90%] rounded-xl rounded-bl-sm bg-slate-100 px-4 py-3 text-sm leading-6 text-slate-800">
                {exchange.question}
              </div>
            </div>
            <div className="flex justify-end">
              <div className="max-w-[90%] rounded-xl rounded-br-sm bg-indigo-700 px-4 py-3 text-sm leading-6 text-white">
                {exchange.answer}
              </div>
            </div>
          </div>
        ))}
        {questions.map((question) => (
          <div key={question.question_id} className="space-y-3">
            <div className="flex justify-start">
              <div className="max-w-[90%] rounded-xl rounded-bl-sm bg-slate-100 px-4 py-3 text-sm leading-6 text-slate-800">
                <p className="font-medium">{question.question}</p>
                <p className="mt-1 text-xs leading-5 text-slate-600">Why this matters: {question.why_it_matters}</p>
              </div>
            </div>
          </div>
        ))}
      </div>

      <form onSubmit={submit} className="space-y-4">
        {questions.map((question) => (
          <label key={question.question_id} className="block space-y-2">
            <span className="text-sm font-medium text-slate-800">Your answer</span>
            <textarea
              required
              minLength={1}
              maxLength={2000}
              rows={3}
              value={answers[question.question_id] ?? ""}
              onChange={(event) => setAnswers((current) => ({
                ...current,
                [question.question_id]: event.target.value,
              }))}
              placeholder="Write a short, specific answer…"
              className="w-full resize-y rounded-lg border border-slate-300 bg-white px-3 py-3 text-sm leading-6 shadow-sm placeholder:text-slate-400"
            />
          </label>
        ))}
        <div className="flex items-center gap-3">
          <button
            type="submit"
            disabled={busy || questions.length === 0 || questions.some((question) => !answers[question.question_id]?.trim())}
            className="rounded-lg bg-indigo-700 px-4 py-2.5 text-sm font-semibold text-white shadow-sm hover:bg-indigo-800 disabled:cursor-not-allowed disabled:bg-slate-300"
          >
            {busy ? "Sending answers…" : "Send answers"}
          </button>
          <span className="text-xs text-slate-500">The workflow remains paused until answers are submitted.</span>
        </div>
      </form>
    </section>
  );
}
