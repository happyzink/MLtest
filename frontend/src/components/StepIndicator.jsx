import { motion } from 'framer-motion';

export default function StepIndicator({ steps, current, onStepClick }) {
  return (
    <div className="flex items-center justify-center gap-2 py-5 px-8">
      {steps.map((label, i) => {
        const isActive = i === current;
        const isDone = i < current;
        return (
          <div key={i} className="flex items-center">
            <button
              onClick={() => onStepClick(i)}
              className="flex items-center gap-2 group"
            >
              <motion.div
                animate={{
                  scale: isActive ? 1.1 : 1,
                  background: isActive
                    ? 'linear-gradient(135deg, #6366f1, #4f46e5)'
                    : isDone
                    ? 'linear-gradient(135deg, #34d399, #059669)'
                    : 'rgba(42,38,64,0.8)',
                }}
                className="w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold text-white border"
                style={{
                  borderColor: isActive ? '#818cf8' : isDone ? '#34d399' : 'rgba(99,102,241,0.2)',
                }}
              >
                {isDone ? '✓' : i + 1}
              </motion.div>
              <span
                className="text-xs font-medium hidden sm:inline"
                style={{ color: isActive ? '#818cf8' : isDone ? '#34d399' : 'var(--text-muted)' }}
              >
                {label}
              </span>
            </button>
            {i < steps.length - 1 && (
              <div
                className="w-8 h-px mx-2"
                style={{ background: isDone ? '#34d399' : 'rgba(99,102,241,0.2)' }}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}
