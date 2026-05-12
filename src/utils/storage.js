import { Platform } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';

const TASK_LOG_KEY = '@mkangel_task_log';
const MAX_LOG_ENTRIES = 50;

export async function getTaskLog() {
  try {
    const json = await AsyncStorage.getItem(TASK_LOG_KEY);
    return json ? JSON.parse(json) : [];
  } catch {
    return [];
  }
}

export async function addTaskLogEntry(entry) {
  try {
    const log = await getTaskLog();
    log.unshift({
      ...entry,
      timestamp: new Date().toISOString(),
    });
    // Keep only the most recent entries
    const trimmed = log.slice(0, MAX_LOG_ENTRIES);
    await AsyncStorage.setItem(TASK_LOG_KEY, JSON.stringify(trimmed));
    return trimmed;
  } catch {
    return [];
  }
}

export async function clearTaskLog() {
  await AsyncStorage.setItem(TASK_LOG_KEY, JSON.stringify([]));
}
