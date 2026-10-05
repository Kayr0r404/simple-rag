# Sync and Encryption

Sync runs every 30 seconds by default. Change the interval with `pebble config set sync.interval 60`.

If two devices edit the same note, Pebble uses last-write-wins conflict resolution. The losing version is saved to the `.conflicts/` folder so nothing is lost.

All notes are end-to-end encrypted with XChaCha20-Poly1305. The encryption key never leaves your device.

Pebble works fully offline and syncs automatically when you reconnect.
