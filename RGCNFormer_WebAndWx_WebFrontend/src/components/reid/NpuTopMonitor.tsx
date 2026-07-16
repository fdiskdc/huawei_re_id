import { useEffect, useState } from 'react';

interface MonitorProcess {
  container: string;
  pid: number;
  npuMemory: number;
  aiCore: number;
  cpu: number;
  memory: number;
  command: string;
  service: string;
}

interface MonitorSnapshot {
  npuMemory: number;
  npuUtilization: number;
  cpu: number;
  memory: number;
  temperature: number;
  power: number;
  processes: MonitorProcess[];
}

const PROCESS_TEMPLATES = [
  {
    container: 'f3ab55c0c126',
    command: 'celery -A tasks_document worker --loglevel=info',
    service: 'rgcn_worker',
  },
  {
    container: 'f3ab55c0c126',
    command: 'celery -A tasks_image worker --loglevel=info',
    service: 'rgcn_worker',
  },
  {
    container: '5dd0f4c8437a',
    command: 'gunicorn -w 1 -b 0.0.0.0:8000 app:app',
    service: 'rgcn_backend',
  },
  {
    container: '5dd0f4c8437a',
    command: 'python -m uvicorn app:app --port 8000',
    service: 'rgcn_backend',
  },
];

const randomBetween = (minimum: number, maximum: number) =>
  minimum + Math.random() * (maximum - minimum);

const formatUptime = (tick: number) => {
  const totalSeconds = (7 * 60 * 60) + (42 * 60) + 10 + tick * 2;
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  return [hours, minutes, seconds].map((value) => String(value).padStart(2, '0')).join(':');
};

const makeSnapshot = (): MonitorSnapshot => ({
  npuMemory: randomBetween(8, 12),
  npuUtilization: randomBetween(70, 90),
  cpu: randomBetween(10, 20),
  memory: randomBetween(60, 128),
  temperature: Math.round(randomBetween(52, 66)),
  power: Math.round(randomBetween(218, 286)),
  processes: PROCESS_TEMPLATES.map((template, index) => ({
    ...template,
    pid: 18420 + index * 137 + Math.round(randomBetween(0, 69)),
    npuMemory: randomBetween(1.4, 3.1),
    aiCore: randomBetween(14, 30),
    cpu: randomBetween(2.5, 12),
    memory: randomBetween(1.2, 5.8),
  })),
});

function Meter({ value, maximum }: { value: number; maximum: number }) {
  const percentage = Math.min(100, Math.max(0, (value / maximum) * 100));
  return (
    <span className="nputop-meter" aria-hidden="true">
      <span style={{ width: `${percentage}%` }} />
    </span>
  );
}

export default function NpuTopMonitor() {
  const [tick, setTick] = useState(0);
  const [snapshot, setSnapshot] = useState(() => makeSnapshot());

  useEffect(() => {
    const timer = window.setInterval(() => {
      setTick((currentTick) => {
        const nextTick = currentTick + 1;
        setSnapshot(makeSnapshot());
        return nextTick;
      });
    }, 1500);
    return () => window.clearInterval(timer);
  }, []);

  return (
    <section className="nputop" aria-label="NPUTOP Ascend 910B performance monitor">
      <div className="nputop-titlebar">
        <strong>NPUTOP 1.0.0</strong>
        <span>Driver Version: 24.1.0</span>
        <span>CANN Version: 8.0.RC3</span>
        <span className="nputop-live"><i /> LIVE</span>
      </div>

      <div className="nputop-device-head" aria-hidden="true">
        <span>NPU&nbsp; Name</span>
        <span>Bus-Id</span>
        <span>HBM-Usage</span>
        <span>NPU-Util&nbsp; Compute M.</span>
      </div>
      <div className="nputop-device-row">
        <div><b>0</b>&nbsp; Ascend 910B</div>
        <div>0000:81:00.0</div>
        <div>
          <b>{snapshot.npuMemory.toFixed(1)} GiB</b> / 32.0 GiB
          <Meter value={snapshot.npuMemory} maximum={32} />
        </div>
        <div>
          <b>{snapshot.npuUtilization.toFixed(0)}%</b>&nbsp; Default
          <Meter value={snapshot.npuUtilization} maximum={100} />
        </div>
        <div className="nputop-device-detail">N/A&nbsp;&nbsp; {snapshot.temperature}C&nbsp;&nbsp; P0&nbsp;&nbsp; {snapshot.power}W / 350W</div>
      </div>

      <div className="nputop-system-grid">
        <div>
          <span>CPU</span>
          <Meter value={snapshot.cpu} maximum={100} />
          <b>{snapshot.cpu.toFixed(1)}%</b>
        </div>
        <div>
          <span>MEM</span>
          <Meter value={snapshot.memory} maximum={512} />
          <b>{snapshot.memory.toFixed(1)} / 512 GiB</b>
        </div>
        <div><span>UPTIME</span><b>{formatUptime(tick)}</b></div>
        <div><span>LOAD AVG</span><b>4.18&nbsp; 3.92&nbsp; 3.71</b></div>
      </div>

      <div className="nputop-section-title">
        <strong>Processes: 4</strong>
        <span>dc@ascend-node</span>
      </div>
      <div className="nputop-table-wrap">
        <table className="nputop-process-table">
          <thead>
            <tr>
              <th>NPU</th><th>PID</th><th>USER</th><th>NPU-MEM</th><th>%AIC</th>
              <th>%CPU</th><th>%MEM</th><th>COMMAND</th>
            </tr>
          </thead>
          <tbody>
            {snapshot.processes.map((process) => (
              <tr key={`${process.container}-${process.command}`}>
                <td>0</td>
                <td>{process.pid}</td>
                <td>root</td>
                <td>{process.npuMemory.toFixed(1)}G</td>
                <td>{process.aiCore.toFixed(0)}</td>
                <td>{process.cpu.toFixed(1)}</td>
                <td>{process.memory.toFixed(1)}</td>
                <td title={process.command}>
                  <span className="nputop-command">{process.command}</span>
                  <small>{process.container} · {process.service}</small>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="nputop-containers">
        <div><span>f3ab55c0c126</span><span>rgcnformer_webandwx_backend-worker</span><span>“celery -A tasks_doc…”</span><span>Up</span><span>8000/tcp</span><b>rgcn_worker</b></div>
        <div><span>5dd0f4c8437a</span><span>rgcnformer_webandwx_backend-backend</span><span>“gunicorn -w 1 -b 0.…”</span><span>Up</span><span>0.0.0.0:8005→8000/tcp</span><b>rgcn_backend</b></div>
      </div>
    </section>
  );
}
