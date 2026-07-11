use std::io::Write;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use cpal::{
    SampleFormat,
    traits::{DeviceTrait as _, HostTrait as _, StreamTrait as _},
};
use ringbuf::traits::{Consumer as _, Producer as _, Split as _};
use rubato::{Fft, FixedSync, Resampler as _, audioadapter_buffers::direct::InterleavedSlice};
use transcribe_cpp::{CommitPolicy, Model, RunOptions, StreamOptions};

fn ensure_backends() {
    transcribe_cpp::init_backends_default().expect("init_backends_default");
}

pub fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .find(|p| p.join("include/transcribe.h").is_file() && p.join("ggml").is_dir())
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."))
}

pub fn streaming_model_path(arg: Option<&str>) -> Option<PathBuf> {
    ensure_backends();
    resolve(
        arg,
        "TRANSCRIBE_STREAMING_MODEL",
        "model/nemotron-3.5-asr-streaming-0.6b-F32.gguf",
    )
}

fn resolve(arg: Option<&str>, env: &str, default_rel: &str) -> Option<PathBuf> {
    let path = arg
        .map(PathBuf::from)
        .or_else(|| std::env::var_os(env).map(PathBuf::from))
        .unwrap_or_else(|| repo_root().join(default_rel));
    path.is_file().then_some(path)
}

/// Downmix interleaved multi-channel f32 to mono by averaging channels.
/// `input.len()` must be a multiple of `channels`.
fn downmix_to_mono(input: &[f32], channels: usize, out: &mut Vec<f32>) {
    if channels == 1 {
        out.extend_from_slice(input); // already mono
        return;
    }
    let inv = 1.0 / channels as f32;
    for frame in input.chunks_exact(channels) {
        let sum: f32 = frame.iter().sum();
        out.push(sum * inv);
    }
}

pub struct Resampler16k {
    inner: Fft<f32>,
    // accumulates mono input until a full chunk is available
    queue: Vec<f32>,
    // reused output scratch, sized to the per-call maximum
    out_buf: Vec<f32>,
}

impl Resampler16k {
    pub fn new(input_rate: u32) -> Self {
        let chunk = 512;
        let inner = Fft::<f32>::new(input_rate as usize, 16_000, chunk, 1, FixedSync::Input)
            .expect("build FFT resampler");

        let out_frames_max = inner.output_frames_max(); // mono => frames == samples
        Resampler16k {
            inner,
            queue: Vec::new(),
            out_buf: vec![0.0; out_frames_max],
        }
    }

    pub fn process(&mut self, input: &[f32], out: &mut Vec<f32>) {
        self.queue.extend_from_slice(input);

        while self.queue.len() >= self.inner.input_frames_next() {
            let needed = self.inner.input_frames_next();

            let written = {
                let in_adapter = InterleavedSlice::new(&self.queue[..needed], 1, needed).unwrap();
                let cap = self.out_buf.len();
                let mut out_adapter = InterleavedSlice::new_mut(&mut self.out_buf, 1, cap).unwrap();

                let (_read, w) = self
                    .inner
                    .process_into_buffer(&in_adapter, &mut out_adapter, None)
                    .unwrap();
                w
            };

            out.extend_from_slice(&self.out_buf[..written]);
            self.queue.drain(..needed);
        }
    }
}

pub struct Caption {
    // byte offset into committed where this line begins
    line_start: usize,
    // last seen committed length (append-only)
    committed_len: usize,
    last_change: Instant,
    // current line already finalized by a pause
    flushed: bool,
    // last thing written (skip redundant writes)
    displayed: String,
    // file the overlay reads (e.g. OBS text source)
    out_path: PathBuf,
    // pause length that ends a line
    silence_flush: Duration,
    // longer pause that blanks the overlay
    clear_after: Duration,
    // cap the visible tail
    max_chars: usize,
}

impl Caption {
    pub fn new(out_path: impl Into<PathBuf>) -> Self {
        Caption {
            line_start: 0,
            committed_len: 0,
            last_change: Instant::now(),
            flushed: false,
            displayed: String::new(),
            out_path: out_path.into(),
            silence_flush: Duration::from_secs_f32(1.5),
            clear_after: Duration::from_secs(4),
            max_chars: 84,
        }
    }

    /// Feed the latest decoder text. Call only when it changed.
    pub fn update(&mut self, committed: &str, tentative: &str) {
        self.committed_len = committed.len();
        // line_start was advanced at the last flush, so this slice is only the
        // text since the last pause. `get` guards against any boundary weirdness.
        let line_committed = committed.get(self.line_start..).unwrap_or("");

        let mut line = String::new();
        line.push_str(line_committed.trim_start());
        let tent = tentative.trim();
        if !tent.is_empty() {
            if !line.is_empty() {
                line.push(' ');
            }
            line.push_str(tent);
        }

        self.flushed = false;
        self.last_change = Instant::now();
        self.render(tail(&line, self.max_chars));
    }

    /// Call every loop iteration to handle pause-based flushing.
    pub fn tick(&mut self) {
        if self.displayed.is_empty() {
            return;
        }
        let idle = self.last_change.elapsed();
        if !self.flushed && idle >= self.silence_flush {
            self.segment_break(); // finalize; keep it on screen for now
        }
        if idle >= self.clear_after {
            self.render(String::new()); // blank the overlay after a long pause
        }
    }

    pub fn segment_break(&mut self) {
        self.line_start = self.committed_len;
        self.flushed = true;
    }

    fn render(&mut self, line: String) {
        if line == self.displayed {
            return;
        }
        // Atomic-ish write so OBS never reads a half-written file.
        let tmp = self.out_path.with_extension("txt.tmp");
        if let Ok(mut f) = std::fs::File::create(&tmp) {
            let _ = f.write_all(line.as_bytes());
            let _ = std::fs::rename(&tmp, &self.out_path);
        }
        // Also overwrite the current terminal line.
        print!("\r\x1b[2K{line}");
        let _ = std::io::stdout().flush();
        self.displayed = line;
    }
}

/// Keep the last `max_chars`, breaking on whole words.
fn tail(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    let mut out: Vec<&str> = Vec::new();
    let mut len = 0;
    for w in s.split_whitespace().rev() {
        let add = w.chars().count() + if out.is_empty() { 0 } else { 1 };
        if len + add > max_chars {
            break;
        }
        len += add;
        out.push(w);
    }
    out.reverse();
    if out.is_empty() {
        s.chars().skip(s.chars().count() - max_chars).collect() // one huge token
    } else {
        out.join(" ")
    }
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);

    let Some(model_path) = streaming_model_path(args.next().as_deref()) else {
        eprintln!("skip streaming: model not found (set TRANSCRIBE_SMOKE_STREAMING_MODEL)");
        return Ok(());
    };

    let model = Model::load(&model_path)?;
    if !model.capabilities().supports_streaming {
        eprintln!("{} does not support streaming", model.arch());
        return Ok(());
    }

    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or_else(|| anyhow::anyhow!("no input device"))?;
    let supported = device.default_input_config()?;

    let device_rate = supported.sample_rate();
    let channels = supported.channels() as usize;
    let sample_format = supported.sample_format();
    let stream_config: cpal::StreamConfig = supported.into();

    println!("input: {device_rate} Hz, {channels} ch, {sample_format:?}  ->  16000 Hz mono");

    let rb = ringbuf::HeapRb::<f32>::new(48_000);
    let (mut prod, mut cons) = rb.split();

    let mut mono = Vec::<f32>::with_capacity(2048);

    let audio_stream = match sample_format {
        SampleFormat::F32 => device.build_input_stream(
            stream_config,
            move |data: &[f32], _: &_| {
                mono.clear();
                downmix_to_mono(data, channels, &mut mono);
                let pushed = prod.push_slice(&mono);
                if pushed < mono.len() {
                    eprintln!("input ring full: dropped {} samples", mono.len() - pushed);
                }
            },
            move |err| eprintln!("stream error: {err}"),
            None,
        )?,
        other => {
            return Err(anyhow::anyhow!("unsupported sample format {other:?}"));
        }
    };

    let mut resampler = Resampler16k::new(device_rate);

    let mut session = model.session()?;
    let opts = StreamOptions {
        commit_policy: CommitPolicy::Auto,
        ..Default::default()
    };
    let mut decoder = session.stream(&RunOptions::default(), &opts)?;
    println!(
        "commit policy: {:?} | initial state: {:?}",
        opts.commit_policy,
        decoder.state()
    );

    audio_stream.play()?;
    println!("listening...");

    const FEED_SAMPLES: usize = 3200; // 200 ms @ 16 kHz; use 1600 for 100 ms
    let mut raw = vec![0.0f32; 4096]; // popped device-rate mono
    let mut resampled = Vec::<f32>::new(); // 16 kHz mono from the resampler
    let mut pending = Vec::<f32>::with_capacity(FEED_SAMPLES * 2);

    let mut caption = Caption::new("/tmp/subtitles.txt");

    loop {
        let n = cons.pop_slice(&mut raw);
        if n > 0 {
            resampled.clear();
            resampler.process(&raw[..n], &mut resampled);

            pending.extend_from_slice(&resampled);

            while pending.len() >= FEED_SAMPLES {
                let update = decoder.feed(&pending[..FEED_SAMPLES])?;
                if update.committed_changed || update.tentative_changed {
                    let text = decoder.text();
                    caption.update(&text.committed, &text.tentative);
                }
                pending.drain(..FEED_SAMPLES);
            }
        }

        caption.tick();
        if n == 0 {
            std::thread::sleep(Duration::from_millis(5));
        }
    }
}
