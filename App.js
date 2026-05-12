import React, { useCallback, useEffect, useState } from 'react';
import {
  Platform,
  Pressable,
  RefreshControl,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import { StatusBar } from 'expo-status-bar';
import * as BackgroundFetch from 'expo-background-fetch';

import {
  registerBackgroundFetch,
  unregisterBackgroundFetch,
  getBackgroundFetchStatus,
  BACKGROUND_FETCH_TASK,
} from './src/tasks/backgroundFetch';
import { getTaskLog, addTaskLogEntry, clearTaskLog } from './src/utils/storage';
import StatusCard from './src/components/StatusCard';
import TaskLogList from './src/components/TaskLogList';

export default function App() {
  const [bgStatus, setBgStatus] = useState(null);
  const [taskLog, setTaskLog] = useState([]);
  const [refreshing, setRefreshing] = useState(false);

  const refresh = useCallback(async () => {
    const status = await getBackgroundFetchStatus();
    setBgStatus(status);
    const log = await getTaskLog();
    setTaskLog(log);
  }, []);

  useEffect(() => {
    refresh();
    // Poll for updates every 10 seconds while the app is foregrounded
    const interval = setInterval(refresh, 10000);
    return () => clearInterval(interval);
  }, [refresh]);

  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    await refresh();
    setRefreshing(false);
  }, [refresh]);

  const toggleBackgroundFetch = useCallback(async () => {
    if (bgStatus?.isRegistered) {
      await unregisterBackgroundFetch();
    } else {
      await registerBackgroundFetch();
    }
    await refresh();
  }, [bgStatus, refresh]);

  const triggerManualTask = useCallback(async () => {
    try {
      const response = await fetch('https://httpbin.org/uuid');
      const data = await response.json();
      await addTaskLogEntry({
        type: 'manual-trigger',
        status: 'success',
        message: `Manual fetch: ${data.uuid}`,
      });
    } catch (error) {
      await addTaskLogEntry({
        type: 'manual-trigger',
        status: 'error',
        message: error.message,
      });
    }
    await refresh();
  }, [refresh]);

  const handleClearLog = useCallback(async () => {
    await clearTaskLog();
    await refresh();
  }, [refresh]);

  const isRegistered = bgStatus?.isRegistered ?? false;
  const statusAvailable =
    bgStatus?.statusCode === BackgroundFetch.BackgroundFetchStatus.Available;

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar style="light" />

      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.appName}>MKAngel</Text>
        <Text style={styles.tagline}>Background Processing</Text>
      </View>

      <ScrollView
        style={styles.content}
        contentContainerStyle={styles.contentInner}
        refreshControl={
          <RefreshControl
            refreshing={refreshing}
            onRefresh={onRefresh}
            tintColor="#4ecca3"
            colors={['#4ecca3']}
          />
        }
      >
        {/* Status Cards */}
        <Text style={styles.sectionTitle}>Services</Text>

        <StatusCard
          title="Background Fetch"
          subtitle={
            isRegistered
              ? 'Active — OS will wake app periodically'
              : 'Disabled — tap toggle to enable'
          }
          enabled={isRegistered}
          onToggle={toggleBackgroundFetch}
          statusColor={isRegistered ? '#4ecca3' : '#e74c3c'}
        />

        <StatusCard
          title="System Status"
          subtitle={bgStatus ? bgStatus.statusName : 'Checking...'}
          statusColor={statusAvailable ? '#4ecca3' : '#f39c12'}
        />

        {/* Actions */}
        <Text style={styles.sectionTitle}>Actions</Text>

        <View style={styles.buttonRow}>
          <Pressable
            style={({ pressed }) => [
              styles.button,
              styles.primaryButton,
              pressed && styles.buttonPressed,
            ]}
            onPress={triggerManualTask}
          >
            <Text style={styles.buttonText}>Run Task Now</Text>
          </Pressable>

          <Pressable
            style={({ pressed }) => [
              styles.button,
              styles.secondaryButton,
              pressed && styles.buttonPressed,
            ]}
            onPress={handleClearLog}
          >
            <Text style={[styles.buttonText, styles.secondaryButtonText]}>
              Clear Log
            </Text>
          </Pressable>
        </View>

        {/* Task Log */}
        <View style={styles.logHeader}>
          <Text style={styles.sectionTitle}>Activity Log</Text>
          <Text style={styles.logCount}>{taskLog.length} entries</Text>
        </View>

        <TaskLogList log={taskLog} />
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0f0f23',
  },
  header: {
    paddingTop: Platform.OS === 'android' ? 48 : 16,
    paddingHorizontal: 20,
    paddingBottom: 16,
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(78, 204, 163, 0.1)',
  },
  appName: {
    fontSize: 28,
    fontWeight: '800',
    color: '#4ecca3',
    letterSpacing: -0.5,
  },
  tagline: {
    fontSize: 13,
    color: '#7f8c8d',
    marginTop: 2,
  },
  content: {
    flex: 1,
  },
  contentInner: {
    padding: 20,
    paddingBottom: 40,
  },
  sectionTitle: {
    color: '#95a5a6',
    fontSize: 12,
    fontWeight: '700',
    textTransform: 'uppercase',
    letterSpacing: 1,
    marginBottom: 12,
    marginTop: 8,
  },
  buttonRow: {
    flexDirection: 'row',
    gap: 10,
    marginBottom: 24,
  },
  button: {
    flex: 1,
    paddingVertical: 14,
    borderRadius: 12,
    alignItems: 'center',
  },
  primaryButton: {
    backgroundColor: '#4ecca3',
  },
  secondaryButton: {
    backgroundColor: 'transparent',
    borderWidth: 1,
    borderColor: '#2c3e50',
  },
  buttonPressed: {
    opacity: 0.7,
  },
  buttonText: {
    fontWeight: '700',
    fontSize: 14,
    color: '#0f0f23',
  },
  secondaryButtonText: {
    color: '#7f8c8d',
  },
  logHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  logCount: {
    color: '#4a5568',
    fontSize: 12,
  },
});
