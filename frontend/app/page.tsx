'use client';

import { useState } from 'react';

const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001';

type TopPlace = { name: string; success_score: number; rating?: number };
type Result = {
  answer: string;
  parsed: { type?: string; district?: string; language?: string };
  retrieved: {
    n_places?: number;
    avg_success_score?: number;
    district_avg_score_all_types?: number;
    top_places?: TopPlace[];
  };
};

const EXAMPLES = [
  'Où ouvrir un café à Agdal ?',
  'Is it a good idea to open a pharmacy in Hassan?',
  'واش نحل صيدلية فأكدال؟',
];

function formatAnswer(text: string) {
  const lines = text.split('\n').filter((l) => l.trim());
  const sections: { title: string; lines: string[] }[] = [];
  let current: { title: string; lines: string[] } | null = null;

  for (const line of lines) {
    const titleMatch = line.match(/^\*\*(.+?)\*\*\s*$/);
    if (titleMatch) {
      if (current) sections.push(current);
      current = { title: titleMatch[1], lines: [] };
    } else {
      if (!current) current = { title: '', lines: [] };
      current.lines.push(line);
    }
  }
  if (current) sections.push(current);

  const isOpen =
    text.includes('**OPEN**') ||
    text.toUpperCase().includes('OPEN') && sections.some((s) => s.title.includes('Recommandation'));

  return (
    <div className="space-y-5">
      {sections.map((section, i) => {
        const isLast = i === sections.length - 1;
        const isRecommendation =
          section.title.toLowerCase().includes('recommandation');

        if (isRecommendation) {
          return (
            <div
              key={i}
              className="rounded-xl border border-green-200 bg-green-50 px-5 py-4"
            >
              <p className="mb-1 text-xs font-semibold uppercase tracking-wider text-green-700">
                {section.title}
              </p>
              {section.lines.map((line, j) => (
                <p
                  key={j}
                  className="text-sm leading-relaxed text-green-800"
                  dangerouslySetInnerHTML={{
                    __html: line.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>'),
                  }}
                />
              ))}
            </div>
          );
        }

        return (
          <div key={i}>
            {section.title && (
              <h3 className="mb-3 border-b border-slate-100 pb-2 text-base font-semibold text-slate-800">
                {section.title}
              </h3>
            )}
            <div className="space-y-2">
              {section.lines.map((line, j) => {
                const hasBold = /\*\*/.test(line);
                return (
                  <p
                    key={j}
                    className={`text-sm leading-relaxed text-slate-600 ${
                      hasBold ? 'border-l-2 border-slate-200 pl-3' : ''
                    }`}
                    dangerouslySetInnerHTML={{
                      __html: line.replace(
                        /\*\*(.+?)\*\*/g,
                        '<strong class="text-slate-800">$1</strong>'
                      ),
                    }}
                  />
                );
              })}
            </div>
          </div>
        );
      })}
    </div>
  );
}

export default function Home() {
  const [question, setQuestion] = useState(EXAMPLES[0]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState<Result | null>(null);

  async function ask(q?: string) {
    const query = q ?? question;
    if (!query.trim()) return;
    setQuestion(query);
    setLoading(true);
    setError('');
    setResult(null);
    try {
      const res = await fetch(`${API}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: query }),
      });
      if (!res.ok) throw new Error(`API ${res.status}`);
      setResult(await res.json());
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Erreur réseau');
    } finally {
      setLoading(false);
    }
  }

  const p = result?.parsed;
  const r = result?.retrieved;
  const mapUrl =
    p?.district && p?.type
      ? `${API}/map?district=${encodeURIComponent(p.district)}&type=${encodeURIComponent(p.type)}`
      : null;

  return (
    <main className="min-h-screen bg-slate-50 text-slate-900">
      <div className="mx-auto max-w-4xl px-6 py-12">
        <h1 className="text-4xl font-bold tracking-tight">MarketMind</h1>
        <p className="mt-1 text-slate-500">
          Intelligence de localisation commerciale — Rabat
        </p>

        <div className="mt-8 flex gap-2">
          <input
            className="flex-1 rounded-xl border border-slate-300 bg-white px-4 py-3 outline-none focus:border-slate-900"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && ask()}
            placeholder="Ex: Où ouvrir une pharmacie à Hassan ?"
          />
          <button
            onClick={() => ask()}
            disabled={loading}
            className="rounded-xl bg-slate-900 px-6 py-3 font-medium text-white transition hover:bg-slate-700 disabled:opacity-50"
          >
            {loading ? 'Analyse…' : 'Analyser'}
          </button>
        </div>

        <div className="mt-3 flex flex-wrap gap-2">
          {EXAMPLES.map((ex) => (
            <button
              key={ex}
              onClick={() => ask(ex)}
              className="rounded-full border border-slate-200 bg-white px-3 py-1 text-sm text-slate-600 hover:border-slate-400"
            >
              {ex}
            </button>
          ))}
        </div>

        {error && (
          <div className="mt-6 rounded-xl bg-red-50 px-4 py-3 text-red-700">
            ⚠️ {error} — l&apos;API tourne-t-elle sur {API} ?
          </div>
        )}

        {result && (
          <div className="mt-8 space-y-6">
            <section className="rounded-2xl bg-white p-6 shadow-sm">
              <h2 className="mb-4 text-xs font-semibold uppercase tracking-wider text-slate-400">
                Recommandation
              </h2>
              {formatAnswer(result.answer)}
            </section>

            {r && (
              <div className="grid grid-cols-3 gap-4">
                <Stat label="Établissements" value={r.n_places ?? '—'} />
                <Stat label="Score moyen" value={r.avg_success_score ?? '—'} />
                <Stat
                  label="Moy. quartier"
                  value={r.district_avg_score_all_types ?? '—'}
                />
              </div>
            )}

            {r?.top_places && r.top_places.length > 0 && (
              <section className="rounded-2xl bg-white p-6 shadow-sm">
                <h2 className="mb-3 text-xs font-semibold uppercase tracking-wider text-slate-400">
                  Meilleurs établissements
                </h2>
                <ul>
                  {r.top_places.map((tp, i) => (
                    <li
                      key={i}
                      className="flex justify-between border-b border-slate-100 py-2 last:border-0"
                    >
                      <span className="text-sm text-slate-700">{tp.name}</span>
                      <span className="font-mono text-sm text-slate-500">
                        {tp.success_score.toFixed(1)}
                      </span>
                    </li>
                  ))}
                </ul>
              </section>
            )}

            {mapUrl && (
              <section className="overflow-hidden rounded-2xl bg-white p-2 shadow-sm">
                <iframe
                  src={mapUrl}
                  className="h-[480px] w-full rounded-xl border-0"
                  title="Carte"
                />
              </section>
            )}
          </div>
        )}
      </div>
    </main>
  );
}

function Stat({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="rounded-2xl bg-white p-5 text-center shadow-sm">
      <div className="text-3xl font-bold">{value}</div>
      <div className="mt-1 text-sm text-slate-500">{label}</div>
    </div>
  );
}