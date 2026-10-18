"use client";

import { FormEvent, useState } from "react";

export interface ProjectDraft {
  name: string;
  raw_brief_text: string;
  rubric_text: string;
}

interface ProjectCreateFormProps {
  busy: boolean;
  onCreate: (draft: ProjectDraft) => Promise<void>;
}

export function ProjectCreateForm({ busy, onCreate }: ProjectCreateFormProps) {
  const [name, setName] = useState("");
  const [brief, setBrief] = useState("");
  const [rubric, setRubric] = useState("");
  const [briefFile, setBriefFile] = useState("");
  const [rubricFile, setRubricFile] = useState("");
  const [fileError, setFileError] = useState("");

  async function loadTextFile(file: File, target: "brief" | "rubric") {
    setFileError("");
    try {
      const contents = await file.text();
      if (target === "brief") {
        setBrief(contents.slice(0, 20_000));
        setBriefFile(file.name);
      } else {
        setRubric(contents.slice(0, 20_000));
        setRubricFile(file.name);
      }
    } catch {
      setFileError(`Could not read ${file.name}. Choose a UTF-8 text file.`);
    }
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (brief.trim().length < 20) return;
    await onCreate({
      name: name.trim() || brief.trim().split("\n")[0].slice(0, 100) || "New project",
      raw_brief_text: brief.trim(),
      rubric_text: rubric.trim(),
    });
  }

  return (
    <form onSubmit={submit} className="space-y-6">
      <div>
        <p className="text-xs font-semibold uppercase tracking-[0.12em] text-indigo-700">New workspace</p>
        <h1 className="mt-2 text-2xl font-semibold tracking-tight text-slate-900">Create a project</h1>
        <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-600">
          Add the assignment brief and rubric. The Requirement Agent will ask for clarification if a critical detail is missing.
        </p>
      </div>

      <label className="block space-y-2">
        <span className="text-sm font-medium text-slate-800">Project name</span>
        <input
          value={name}
          onChange={(event) => setName(event.target.value)}
          maxLength={200}
          placeholder="e.g. Rainfall analysis"
          className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2.5 text-sm shadow-sm placeholder:text-slate-400"
        />
      </label>

      <div className="grid gap-5 xl:grid-cols-2">
        <label className="block space-y-2">
          <span className="flex items-center justify-between gap-4 text-sm font-medium text-slate-800">
            <span>Assignment brief <span className="text-rose-600">*</span></span>
            <span className="text-xs font-normal text-slate-500">{brief.length.toLocaleString()} / 20,000</span>
          </span>
          <textarea
            required
            minLength={20}
            maxLength={20_000}
            rows={14}
            value={brief}
            onChange={(event) => setBrief(event.target.value)}
            placeholder="Paste the project brief here, or upload a .txt, .md, or .json file."
            className="w-full resize-y rounded-lg border border-slate-300 bg-white px-3 py-3 text-sm leading-6 shadow-sm placeholder:text-slate-400"
          />
          <span className="flex flex-wrap items-center gap-3 text-xs text-slate-500">
            <input
              aria-label="Upload brief file"
              type="file"
              accept=".txt,.md,.markdown,.json,text/plain,text/markdown,application/json"
              onChange={(event) => {
                const file = event.target.files?.[0];
                if (file) void loadTextFile(file, "brief");
              }}
              className="max-w-full text-xs file:mr-3 file:rounded-md file:border-0 file:bg-slate-100 file:px-3 file:py-1.5 file:text-xs file:font-medium file:text-slate-700 hover:file:bg-slate-200"
            />
            {briefFile && <span>Loaded {briefFile}</span>}
          </span>
        </label>

        <label className="block space-y-2">
          <span className="flex items-center justify-between gap-4 text-sm font-medium text-slate-800">
            <span>Rubric <span className="font-normal text-slate-500">(optional)</span></span>
            <span className="text-xs font-normal text-slate-500">{rubric.length.toLocaleString()} / 20,000</span>
          </span>
          <textarea
            maxLength={20_000}
            rows={14}
            value={rubric}
            onChange={(event) => setRubric(event.target.value)}
            placeholder="Paste the marking criteria, or upload a text file."
            className="w-full resize-y rounded-lg border border-slate-300 bg-white px-3 py-3 text-sm leading-6 shadow-sm placeholder:text-slate-400"
          />
          <span className="flex flex-wrap items-center gap-3 text-xs text-slate-500">
            <input
              aria-label="Upload rubric file"
              type="file"
              accept=".txt,.md,.markdown,.json,text/plain,text/markdown,application/json"
              onChange={(event) => {
                const file = event.target.files?.[0];
                if (file) void loadTextFile(file, "rubric");
              }}
              className="max-w-full text-xs file:mr-3 file:rounded-md file:border-0 file:bg-slate-100 file:px-3 file:py-1.5 file:text-xs file:font-medium file:text-slate-700 hover:file:bg-slate-200"
            />
            {rubricFile && <span>Loaded {rubricFile}</span>}
          </span>
        </label>
      </div>

      {fileError && <p role="alert" className="text-sm text-rose-700">{fileError}</p>}

      <div className="flex flex-wrap items-center gap-4 border-t border-slate-200 pt-5">
        <button
          type="submit"
          disabled={busy || brief.trim().length < 20}
          className="inline-flex items-center justify-center rounded-lg bg-indigo-700 px-4 py-2.5 text-sm font-semibold text-white shadow-sm transition hover:bg-indigo-800 disabled:cursor-not-allowed disabled:bg-slate-300"
        >
          {busy ? "Creating project…" : "Create project"}
        </button>
        <span className="text-xs text-slate-500">Briefs and rubrics are limited to 20,000 characters each.</span>
      </div>
    </form>
  );
}
