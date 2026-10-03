'use strict';

// One shared clock keeps every demonstration on the same playback timeline.
const videos = [...document.querySelectorAll('video')];
const motionButton = document.querySelector('#motion-toggle');
const motionStatus = document.querySelector('#motion-status');
let mediaEpoch = null;
let pausedAt = 0;
let userPaused = false;

function waitForVideo(video) {
  video.muted = true;
  video.defaultMuted = true;
  video.pause();
  if (video.readyState >= 3 || video.error) return Promise.resolve();
  return new Promise(resolve => {
    let timer;
    const done = () => {
      clearTimeout(timer);
      video.removeEventListener('canplay', done);
      video.removeEventListener('error', done);
      resolve();
    };
    video.addEventListener('canplay', done, { once: true });
    video.addEventListener('error', done, { once: true });
    timer = setTimeout(done, 10000);
  });
}

function alignVideo(video, elapsed) {
  if (!Number.isFinite(video.duration) || video.duration <= 0 || video.readyState < 2) return;
  const target = elapsed % video.duration;
  const difference = Math.abs(video.currentTime - target);
  const circularDifference = Math.min(difference, video.duration - difference);
  if (circularDifference > 0.16) video.currentTime = target;
}

async function playVideos() {
  userPaused = false;
  mediaEpoch = performance.now() - pausedAt * 1000;
  videos.forEach(video => alignVideo(video, pausedAt));
  const outcomes = await Promise.allSettled(videos.map(video => video.play()));
  const blocked = outcomes.some(outcome => outcome.status === 'rejected');
  motionButton.textContent = blocked ? 'Play videos' : 'Pause videos';
  motionButton.setAttribute('aria-pressed', 'false');
  motionStatus.textContent = blocked ? 'Use Play videos to start the demonstrations.' : 'All demonstrations are playing.';
}

Promise.all(videos.map(waitForVideo)).then(() => {
  if (!userPaused) {
    pausedAt = 0;
    videos.forEach(video => { if (video.readyState >= 1) video.currentTime = 0; });
    playVideos();
  }
});

motionButton.addEventListener('click', () => {
  if (userPaused || videos.every(video => video.paused) || motionButton.textContent === 'Play videos') {
    playVideos();
  } else {
    userPaused = true;
    pausedAt = mediaEpoch === null ? 0 : (performance.now() - mediaEpoch) / 1000;
    videos.forEach(video => video.pause());
    motionButton.textContent = 'Play videos';
    motionButton.setAttribute('aria-pressed', 'true');
    motionStatus.textContent = 'All demonstrations are paused.';
  }
});

setInterval(() => {
  if (userPaused || mediaEpoch === null || document.hidden) return;
  const elapsed = (performance.now() - mediaEpoch) / 1000;
  videos.forEach(video => {
    alignVideo(video, elapsed);
    if (video.paused && !video.error) video.play().catch(() => {});
  });
}, 1000);

const copyButton = document.querySelector('#copy-citation');
copyButton.addEventListener('click', async () => {
  const citation = document.querySelector('#bibtex').textContent;
  try {
    await navigator.clipboard.writeText(citation);
    copyButton.querySelector('span').textContent = 'Copied';
    document.querySelector('#copy-status').textContent = 'BibTeX copied to clipboard.';
  } catch {
    const range = document.createRange();
    range.selectNodeContents(document.querySelector('#bibtex'));
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    document.querySelector('#copy-status').textContent = 'Citation selected. Press Ctrl+C or Command+C to copy.';
  }
});
