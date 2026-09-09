import {
  BarChart,
  Bar,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

export default function ProbabilityChart({ result }) {
  if (!result) return null;

  const data = Object.entries(result.probabilities).map(([name, value]) => ({
    name,
    probability: Number((value * 100).toFixed(2)),
  }));

  return (
    <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-soft">
      <div className="mb-5">
        <h3 className="text-base font-bold text-slate-900">
          Class probabilities
        </h3>
        <p className="mt-1 text-sm text-slate-500">
          Probability distribution returned by the softmax output layer.
        </p>
      </div>

      <div className="h-[320px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={data}
            margin={{ top: 10, right: 20, left: 0, bottom: 35 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              dataKey="name"
              angle={-20}
              textAnchor="end"
              interval={0}
              height={65}
              tick={{ fontSize: 11 }}
            />
            <YAxis
              domain={[0, 100]}
              tickFormatter={(value) => `${value}%`}
              tick={{ fontSize: 11 }}
            />
            <Tooltip formatter={(value) => [`${value}%`, "Probability"]} />
            <Bar dataKey="probability" radius={[8, 8, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
