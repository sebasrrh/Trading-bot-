export const BOOTSTRAP_SHADER = `
struct Params {
  numPaths: u32,
  horizon: u32,
  seed: u32,
  blockLen: u32,
  nSource: u32,
  ruinThreshold: f32,
  ghostCount: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> sourceReturns: array<f32>;
@group(0) @binding(2) var<storage, read_write> terminalValues: array<f32>;
@group(0) @binding(3) var<storage, read_write> maxDdValues: array<f32>;
@group(0) @binding(4) var<storage, read_write> ruinedFlags: array<u32>;
@group(0) @binding(5) var<storage, read_write> fanMatrix: array<f32>;
@group(0) @binding(6) var<storage, read_write> ghostPaths: array<f32>;

fn xorshift(state: ptr<function, u32>) -> u32 {
  var s = *state;
  s = s ^ (s << 13u);
  s = s ^ (s >> 17u);
  s = s ^ (s << 5u);
  *state = s;
  return s;
}

fn rand(state: ptr<function, u32>) -> f32 {
  return f32(xorshift(state)) / 4294967296.0;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let pathIdx = id.x;
  if (pathIdx >= params.numPaths) { return; }

  var state = params.seed + pathIdx * 2654435761u;
  let n = params.nSource;
  let blk = params.blockLen;
  let H = params.horizon;
  let thresh = params.ruinThreshold;

  var eq: f32 = 1.0;
  var peak: f32 = 1.0;
  var maxDD: f32 = 0.0;
  var ruined: u32 = 0u;

  var i: u32 = 0u;
  while (i < H) {
    let startIdx = u32(rand(&state) * f32(n));
    let remain = H - i;
    var L = blk;
    if (L > remain) { L = remain; }
    for (var j: u32 = 0u; j < L; j = j + 1u) {
      let idx = (startIdx + j) % n;
      let r = sourceReturns[idx];
      eq = eq * exp(r);
      if (eq > peak) { peak = eq; }
      let dd = eq / peak - 1.0;
      if (dd < maxDD) { maxDD = dd; }
      if (eq < thresh && ruined == 0u) { ruined = 1u; }
      fanMatrix[(i + j) * params.numPaths + pathIdx] = eq;
    }
    i = i + L;
  }

  terminalValues[pathIdx] = eq;
  maxDdValues[pathIdx] = maxDD;
  ruinedFlags[pathIdx] = ruined;

  if (pathIdx < params.ghostCount) {
    for (var k: u32 = 0u; k < H; k = k + 1u) {
      ghostPaths[pathIdx * H + k] = fanMatrix[k * params.numPaths + pathIdx];
    }
  }
}
`;