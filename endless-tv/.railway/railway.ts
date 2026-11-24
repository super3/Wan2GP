// Endless TV on Railway: one Node service, and a volume for everything the GPU boxes upload (clips, logos, posters, runs).
// Railway's config-as-code file (railway.toml) is no longer read after 2026-12-01, so this is the current format: apply it
// from endless-tv/ with the Railway CLI 5.42.1 or newer (`npm install`, `railway link`, `railway config plan`, then
// `railway config apply`). See ../README.md.
import { defineRailway, github, preserve, project, service, volume } from "railway/iac";

export default defineRailway(() => {
  const media = volume("endless-tv-media", {
    sizeMB: 5120,  // 5 GB, the Hobby plan's most: a 5 s 480p clip is about 0.4 MB, a 30 s one about 2 MB
  });

  const web = service("endless-tv", {
    // The app is on PR #8's branch until that PR merges; then this becomes "main".
    source: github("super3/Wan2GP", { branch: "ccr-fcf4a278-40swa4", rootDirectory: "endless-tv" }),
    build: {
      builder: "RAILPACK",
      watchPatterns: ["/endless-tv/**"],  // from the repo root: changes elsewhere in Wan2GP don't redeploy it
    },
    start: "npm start",
    healthcheck: "/health",
    healthcheckTimeout: 30,
    deploy: {
      restartPolicyType: "ON_FAILURE",
      restartPolicyMaxRetries: 10,
    },
    volumeMounts: {
      "/data": media,  // the app finds it through RAILWAY_VOLUME_MOUNT_PATH
    },
    env: {
      INGEST_TOKEN: preserve(),  // the upload token: set it in Railway (never here); applying keeps its value
      KEEP_CLIPS: "0",           // clips kept per channel, oldest dropped first; 0 keeps them all
      MAX_UPLOAD_MB: "300",
    },
    // Railway's edge cache keeps clips, logos and posters (served as immutable) off the app's bandwidth. It needs a
    // public domain: generate one in the service's settings first, then uncomment and apply again.
    // cdn: { enabled: true },
  });

  return project("endless-tv", { resources: [web, media] });
});
