import { useState, useRef, useCallback, useEffect } from "react";
import type { NewOrder } from "@tradeboard/paper";
import { usePaperStore } from "../state/paper-store";
import { useContextStore } from "../state/context-store";

export default function PaperTrading() {
  const ctx = useContextStore();
  const sym = ctx.channels.A.symbol;
  const getEngine = usePaperStore((s) => s.getEngine);
  const bump = usePaperStore((s) => s.bump);
  const refreshKey = usePaperStore((s) => s.refreshKey);
  const eng = getEngine();
  void refreshKey;

  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState("");
  const [orderSide, setOrderSide] = useState<"buy" | "sell">("buy");
  const [orderQty, setOrderQty] = useState(10);
  const [orderType, setOrderType] = useState<"market" | "limit" | "stop">("market");
  const [orderPrice, setOrderPrice] = useState("");
  const [bindings, setBindings] = useState(eng.account.autoStrategies);
  const [autoFeed, setAutoFeed] = useState(false);
  const [autoInterval, setAutoInterval] = useState(10);
  const fetchInProgress = useRef(false);
  const autoRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const autoEnabled = useRef(false);

  // Keep ref in sync
  autoEnabled.current = autoFeed;

  const doFeed = useCallback(async () => {
    if (fetchInProgress.current) return;
    fetchInProgress.current = true;
    if (!autoEnabled.current) setRunning(true);
    setProgress("Fetching bars\u2026");
    try {
      const res = await fetch(`http://localhost:8787/api/bars?symbol=${sym}&timeframe=1D&from=0&to=${Date.now()}`);
      const data = await res.json();
      const arr = data.bars ?? [];
      if (arr.length === 0) { setProgress("No data"); setRunning(false); fetchInProgress.current = false; return; }
      const bars = arr.map((b: any) => ({ o: b.o, h: b.h, l: b.l, c: b.c, v: b.v ?? 0, t: b.t }));
      eng.processBars(sym, bars);
      setProgress(`\u2713 ${arr.length} bars`);
      bump();
    } catch (err) { setProgress(`Error: ${err}`); }
    if (!autoEnabled.current) setRunning(false);
    fetchInProgress.current = false;
  }, [sym, eng, bump]);

  // Auto-feed interval
  useEffect(() => {
    if (autoFeed) {
      autoRef.current = setInterval(() => { doFeed(); }, autoInterval * 1000);
    }
    return () => {
      if (autoRef.current) { clearInterval(autoRef.current); autoRef.current = null; }
    };
  }, [autoFeed, autoInterval, doFeed]);

  const placeOrder = useCallback(() => {
    const n: NewOrder = { symbol: sym, side: orderSide, qty: orderQty, type: orderType };
    if ((orderType === "limit" || orderType === "stop") && orderPrice) {
      if (orderType === "limit") n.limitPrice = Number(orderPrice);
      else n.stopPrice = Number(orderPrice);
    }
    eng.placeOrder(n);
    bump();
  }, [sym, orderSide, orderQty, orderType, orderPrice, eng, bump]);

  const cancelOrder = useCallback((id: string) => {
    eng.cancelOrder(id);
    bump();
  }, [eng, bump]);

  const toggleBinding = useCallback((id: string) => {
    const b = eng.account.autoStrategies.find((x) => x.id === id);
    if (b) { b.enabled = !b.enabled; setBindings([...eng.account.autoStrategies]); bump(); }
  }, [eng, bump]);

  const eq = eng.equity();
  const positions = [...eng.account.positions.values()];
  const openOrders = eng.account.openOrders.filter((o) => o.status === "open");
  const fills = eng.account.fills.slice(-50).reverse();

  return (
    <div style={{ flex: 1, display: "flex", overflow: "hidden" }}>
      <div style={{ width: 300, borderRight: "1px solid var(--border-hairline)", padding: 12, overflow: "auto", display: "flex", flexDirection: "column", gap: 12 }}>
        <h2 style={{ fontSize: 14, fontWeight: 700, margin: 0 }}>Paper Trading</h2>
        <div style={{ background: "var(--bg-surface-1)", borderRadius: "var(--radius-s)", padding: 10, display: "flex", flexDirection: "column", gap: 4 }}>
          <div style={{ fontSize: 11, color: "var(--text-muted)" }}>ACCOUNT</div>
          <Row label="Cash" val={`$${eq.cash.toLocaleString("en-US", { minimumFractionDigits: 2 })}`} />
          <Row label="Equity" val={`$${eq.total.toLocaleString("en-US", { minimumFractionDigits: 2 })}`} />
          <Row label="Return" val={`${(eq.ret * 100).toFixed(2)}%`} color={eq.ret >= 0 ? "var(--accent)" : "var(--loss)"} />
          <Row label="Positions" val={`${positions.length}`} />
          <Row label="Open Orders" val={`${openOrders.length}`} />
        </div>

        <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
          <button onClick={doFeed} disabled={running || autoFeed}
            style={{ flex: 1, padding: "8px 16px", border: "none", borderRadius: "var(--radius-s)", background: autoFeed ? "var(--bg-surface-2)" : "var(--accent)", color: autoFeed ? "var(--text-muted)" : "#fff", fontSize: 13, fontWeight: 600, cursor: "pointer" }}>
            {autoFeed ? `Auto ${autoInterval}s` : running ? progress : `Feed ${sym} Bars`}
          </button>
          <label style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 11, color: "var(--text-secondary)", cursor: "pointer" }}>
            <input type="checkbox" checked={autoFeed} onChange={(e) => setAutoFeed(e.target.checked)} />
            Auto
          </label>
        </div>
        {autoFeed && (
          <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <span style={{ fontSize: 11, color: "var(--text-muted)" }}>Every</span>
            <input type="number" value={autoInterval} min={5} max={300}
              onChange={(e) => setAutoInterval(Number(e.target.value))}
              style={{ width: 60, padding: "3px 4px", border: "1px solid var(--border-hairline)", borderRadius: "var(--radius-s)", background: "var(--bg-surface-2)", color: "var(--text-primary)", fontSize: 11 }} />
            <span style={{ fontSize: 11, color: "var(--text-muted)" }}>sec</span>
          </div>
        )}

        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          <label style={{ fontSize: 11, color: "var(--text-muted)", textTransform: "uppercase" }}>Place Order</label>
          <div style={{ display: "flex", gap: 4 }}>
            <button onClick={() => setOrderSide("buy")}
              style={{ flex: 1, padding: "4px 6px", border: orderSide === "buy" ? "1px solid var(--accent)" : "1px solid var(--border-hairline)", borderRadius: "var(--radius-s)", background: orderSide === "buy" ? "var(--accent-soft)" : "var(--bg-surface-2)", color: orderSide === "buy" ? "var(--accent)" : "var(--text-secondary)", fontSize: 11, cursor: "pointer" }}>Buy</button>
            <button onClick={() => setOrderSide("sell")}
              style={{ flex: 1, padding: "4px 6px", border: orderSide === "sell" ? "1px solid var(--loss)" : "1px solid var(--border-hairline)", borderRadius: "var(--radius-s)", background: orderSide === "sell" ? "var(--loss-soft)" : "var(--bg-surface-2)", color: orderSide === "sell" ? "var(--loss)" : "var(--text-secondary)", fontSize: 11, cursor: "pointer" }}>Sell</button>
          </div>
          <select value={orderType} onChange={(e) => setOrderType(e.target.value as any)}
            style={{ padding: "4px 6px", border: "1px solid var(--border-hairline)", borderRadius: "var(--radius-s)", background: "var(--bg-surface-2)", color: "var(--text-primary)", fontSize: 12 }}>
            <option value="market">Market</option>
            <option value="limit">Limit</option>
            <option value="stop">Stop</option>
          </select>
          <input type="number" value={orderQty} onChange={(e) => setOrderQty(Number(e.target.value))} placeholder="Qty"
            style={{ padding: "4px 6px", border: "1px solid var(--border-hairline)", borderRadius: "var(--radius-s)", background: "var(--bg-surface-2)", color: "var(--text-primary)", fontSize: 12 }} />
          {(orderType === "limit" || orderType === "stop") && (
            <input type="number" value={orderPrice} onChange={(e) => setOrderPrice(e.target.value)} placeholder={orderType === "limit" ? "Limit price" : "Stop price"}
              style={{ padding: "4px 6px", border: "1px solid var(--border-hairline)", borderRadius: "var(--radius-s)", background: "var(--bg-surface-2)", color: "var(--text-primary)", fontSize: 12 }} />
          )}
          <button onClick={placeOrder}
            style={{ padding: "6px 12px", border: "none", borderRadius: "var(--radius-s)", background: "var(--accent)", color: "#fff", fontSize: 12, fontWeight: 600, cursor: "pointer" }}>
            Place {orderSide === "buy" ? "Buy" : "Sell"} Order
          </button>
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <label style={{ fontSize: 11, color: "var(--text-muted)", textTransform: "uppercase" }}>Auto Strategies</label>
          {bindings.map((b) => (
            <div key={b.id} style={{ display: "flex", alignItems: "center", gap: 6, padding: "4px 6px", background: "var(--bg-surface-1)", borderRadius: "var(--radius-s)", fontSize: 11 }}>
              <input type="checkbox" checked={b.enabled} onChange={() => toggleBinding(b.id)} style={{ cursor: "pointer" }} />
              <div style={{ flex: 1 }}>
                <div style={{ fontWeight: 600 }}>{b.strategyId}</div>
                <div style={{ color: "var(--text-muted)" }}>{b.symbol} \u00B7 {(b.allocation * 100).toFixed(0)}%</div>
              </div>
            </div>
          ))}
        </div>
      </div>
      <div style={{ flex: 1, padding: 12, overflow: "auto", display: "flex", flexDirection: "column", gap: 12 }}>
        <div>
          <h3 style={{ fontSize: 13, fontWeight: 600, margin: "0 0 4px 0" }}>Positions ({positions.length})</h3>
          {positions.length === 0 ? (
            <div style={{ color: "var(--text-muted)", fontSize: 12, padding: 8 }}>No open positions.</div>
          ) : (
            <Table cols={["Symbol", "Qty", "Avg Price", "Unrealized P&L"]}>
              {positions.map((p) => (
                <tr key={p.symbol} style={{ borderBottom: "1px solid var(--border-hairline)" }}>
                  <td style={{ padding: "4px 8px", fontWeight: 600 }}>{p.symbol}</td>
                  <td style={{ padding: "4px 8px", textAlign: "right" }}>{p.qty}</td>
                  <td style={{ padding: "4px 8px", textAlign: "right" }}>${p.avgPrice.toFixed(2)}</td>
                  <td style={{ padding: "4px 8px", textAlign: "right", color: p.unrealizedPnl >= 0 ? "var(--accent)" : "var(--loss)", fontWeight: 600 }}>${p.unrealizedPnl.toFixed(2)}</td>
                </tr>
              ))}
            </Table>
          )}
        </div>
        <div>
          <h3 style={{ fontSize: 13, fontWeight: 600, margin: "0 0 4px 0" }}>Open Orders ({openOrders.length})</h3>
          {openOrders.length === 0 ? (
            <div style={{ color: "var(--text-muted)", fontSize: 12, padding: 8 }}>No open orders.</div>
          ) : (
            <Table cols={["Symbol", "Side", "Qty", "Type", "Price", ""]}>
              {openOrders.map((o) => (
                <tr key={o.id} style={{ borderBottom: "1px solid var(--border-hairline)" }}>
                  <td style={{ padding: "4px 8px", fontWeight: 600 }}>{o.symbol}</td>
                  <td style={{ padding: "4px 8px", color: o.side === "buy" ? "var(--accent)" : "var(--loss)" }}>{o.side}</td>
                  <td style={{ padding: "4px 8px", textAlign: "right" }}>{o.qty}</td>
                  <td style={{ padding: "4px 8px" }}>{o.type}</td>
                  <td style={{ padding: "4px 8px", textAlign: "right" }}>{o.limitPrice ?? o.stopPrice ?? "\u2014"}</td>
                  <td style={{ padding: "4px 8px", textAlign: "right" }}>
                    <button onClick={() => cancelOrder(o.id)}
                      style={{ border: "none", background: "transparent", color: "var(--loss)", cursor: "pointer", fontSize: 11, textDecoration: "underline" }}>Cancel</button>
                  </td>
                </tr>
              ))}
            </Table>
          )}
        </div>
        <div>
          <h3 style={{ fontSize: 13, fontWeight: 600, margin: "0 0 4px 0" }}>Fill History ({fills.length})</h3>
          {fills.length === 0 ? (
            <div style={{ color: "var(--text-muted)", fontSize: 12, padding: 8 }}>No fills yet.</div>
          ) : (
            <div style={{ maxHeight: 300, overflow: "auto" }}>
              <Table cols={["Time", "Symbol", "Side", "Qty", "Price", "P&L"]}>
                {fills.map((f) => (
                  <tr key={f.id} style={{ borderBottom: "1px solid var(--border-hairline)" }}>
                    <td style={{ padding: "3px 6px", color: "var(--text-muted)" }}>{new Date(f.t).toLocaleDateString()}</td>
                    <td style={{ padding: "3px 6px", fontWeight: 600 }}>{f.symbol}</td>
                    <td style={{ padding: "3px 6px", color: f.side === "buy" ? "var(--accent)" : "var(--loss)" }}>{f.side}</td>
                    <td style={{ padding: "3px 6px", textAlign: "right" }}>{f.qty}</td>
                    <td style={{ padding: "3px 6px", textAlign: "right" }}>${f.price.toFixed(2)}</td>
                    <td style={{ padding: "3px 6px", textAlign: "right", color: f.pnl != null ? (f.pnl >= 0 ? "var(--accent)" : "var(--loss)") : "var(--text-muted)", fontWeight: 600 }}>
                      {f.pnl != null ? `$${f.pnl.toFixed(2)}` : "\u2014"}
                    </td>
                  </tr>
                ))}
              </Table>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function Row({ label, val, color }: { label: string; val: string; color?: string }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12 }}>
      <span style={{ color: "var(--text-secondary)" }}>{label}</span>
      <span style={{ fontWeight: 600, color }}>{val}</span>
    </div>
  );
}

function Table({ cols, children }: { cols: string[]; children: React.ReactNode }) {
  return (
    <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
      <thead><tr style={{ borderBottom: "2px solid var(--border-hairline)" }}>
        {cols.map((c) => <th key={c} style={{ textAlign: "left", padding: "4px 8px" }}>{c}</th>)}
      </tr></thead>
      <tbody>{children}</tbody>
    </table>
  );
}