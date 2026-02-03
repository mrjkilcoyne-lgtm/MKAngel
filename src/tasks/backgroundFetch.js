import * as BackgroundFetch from 'expo-background-fetch';
import * as TaskManager from 'expo-task-manager';
import { addTaskLogEntry } from '../utils/storage';

export const BACKGROUND_FETCH_TASK = 'MKANGEL_BACKGROUND_FETCH';

// Define the background task
TaskManager.defineTask(BACKGROUND_FETCH_TASK, async () => {
  const now = new Date();
  console.log(
    `[MKAngel] Background fetch fired at ${now.toISOString()}`
  );

  try {
    // Simulate background work — replace with your real logic
    const result = await performBackgroundWork();

    await addTaskLogEntry({
      type: 'background-fetch',
      status: 'success',
      message: result,
    });

    return BackgroundFetch.BackgroundFetchResult.NewData;
  } catch (error) {
    await addTaskLogEntry({
      type: 'background-fetch',
      status: 'error',
      message: error.message,
    });

    return BackgroundFetch.BackgroundFetchResult.Failed;
  }
});

async function performBackgroundWork() {
  // Example: fetch a random fact as proof of background activity
  try {
    const response = await fetch('https://httpbin.org/uuid');
    const data = await response.json();
    return `Fetched UUID: ${data.uuid}`;
  } catch {
    // Fallback if network isn't available
    return `Background tick at ${new Date().toLocaleTimeString()}`;
  }
}

export async function registerBackgroundFetch() {
  const status = await BackgroundFetch.getStatusAsync();

  if (
    status === BackgroundFetch.BackgroundFetchStatus.Restricted ||
    status === BackgroundFetch.BackgroundFetchStatus.Denied
  ) {
    console.log('[MKAngel] Background fetch is restricted or denied');
    return false;
  }

  const isRegistered = await TaskManager.isTaskRegisteredAsync(
    BACKGROUND_FETCH_TASK
  );

  if (!isRegistered) {
    await BackgroundFetch.registerTaskAsync(BACKGROUND_FETCH_TASK, {
      minimumInterval: 15 * 60, // 15 minutes (OS minimum)
      stopOnTerminate: false,
      startOnBoot: true,
    });
    console.log('[MKAngel] Background fetch registered');
  }

  return true;
}

export async function unregisterBackgroundFetch() {
  const isRegistered = await TaskManager.isTaskRegisteredAsync(
    BACKGROUND_FETCH_TASK
  );

  if (isRegistered) {
    await BackgroundFetch.unregisterTaskAsync(BACKGROUND_FETCH_TASK);
    console.log('[MKAngel] Background fetch unregistered');
  }
}

export async function getBackgroundFetchStatus() {
  const status = await BackgroundFetch.getStatusAsync();
  const isRegistered = await TaskManager.isTaskRegisteredAsync(
    BACKGROUND_FETCH_TASK
  );

  return {
    statusCode: status,
    statusName: BackgroundFetch.BackgroundFetchStatus[status] || 'Unknown',
    isRegistered,
  };
}
