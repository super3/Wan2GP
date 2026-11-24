// Starts Endless TV. On Railway, the volume's mount path (RAILWAY_VOLUME_MOUNT_PATH) holds everything uploaded; without
// a volume the files live in ./data and are lost at the next deploy.
const path = require('path');
const { Store } = require('./store');
const { createApp } = require('./app');

function config(env = process.env) {
  const volume = env.RAILWAY_VOLUME_MOUNT_PATH;
  return {
    port: Number(env.PORT) || 3000,
    dataDir: path.resolve(env.DATA_DIR || volume || 'data'),  // ./data: the app folder, for `npm start` and on Railway
    persistent: Boolean(env.DATA_DIR || volume),
    token: env.INGEST_TOKEN || '',
    maxUploadBytes: (Number(env.MAX_UPLOAD_MB) || 300) * 1e6,
    keepClips: Number(env.KEEP_CLIPS) || 0,
    discordUrl: env.DISCORD_URL || '',  // the invite the header's Discord button opens
  };
}

async function start(env) {
  const settings = config(env);
  const store = await new Store(settings.dataDir, { keepClips: settings.keepClips }).init();
  const app = createApp({ store, token: settings.token, maxUploadBytes: settings.maxUploadBytes, persistent: settings.persistent, discordUrl: settings.discordUrl });
  const server = app.listen(settings.port, () => {
    console.log(`Endless TV on port ${server.address().port}, data in ${settings.dataDir}${settings.persistent ? '' : ' (not persistent)'}, uploads ${settings.token ? 'on' : 'off (no INGEST_TOKEN)'}`);
  });
  return server;
}

/* istanbul ignore next -- the entry point when run as `node server/index.js`; start() itself is tested */
if (require.main === module) {
  start(process.env).catch((error) => {
    console.error(error);
    process.exit(1);
  });
}

module.exports = { config, start };
