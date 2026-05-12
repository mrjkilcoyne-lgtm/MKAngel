import React from 'react';
import { FlatList, StyleSheet, Text, View } from 'react-native';

function formatTime(isoString) {
  const d = new Date(isoString);
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

function formatDate(isoString) {
  const d = new Date(isoString);
  return d.toLocaleDateString([], { month: 'short', day: 'numeric' });
}

function LogEntry({ item }) {
  const isError = item.status === 'error';
  return (
    <View style={[styles.entry, isError && styles.entryError]}>
      <View style={styles.entryHeader}>
        <Text style={styles.entryType}>{item.type}</Text>
        <Text style={styles.entryTime}>
          {formatDate(item.timestamp)} {formatTime(item.timestamp)}
        </Text>
      </View>
      <Text style={[styles.entryMessage, isError && styles.errorText]}>
        {item.message}
      </Text>
      <View style={[styles.statusBadge, isError ? styles.badgeError : styles.badgeSuccess]}>
        <Text style={styles.statusText}>{item.status}</Text>
      </View>
    </View>
  );
}

export default function TaskLogList({ log }) {
  if (!log || log.length === 0) {
    return (
      <View style={styles.emptyContainer}>
        <Text style={styles.emptyIcon}>{'{ }'}</Text>
        <Text style={styles.emptyText}>No background tasks yet</Text>
        <Text style={styles.emptySubtext}>
          Enable background fetch and tasks will appear here
        </Text>
      </View>
    );
  }

  return (
    <FlatList
      data={log}
      keyExtractor={(item, index) => `${item.timestamp}-${index}`}
      renderItem={({ item }) => <LogEntry item={item} />}
      style={styles.list}
      contentContainerStyle={styles.listContent}
    />
  );
}

const styles = StyleSheet.create({
  list: {
    flex: 1,
  },
  listContent: {
    paddingBottom: 20,
  },
  entry: {
    backgroundColor: '#1a1a2e',
    borderRadius: 12,
    padding: 14,
    marginBottom: 10,
    borderLeftWidth: 3,
    borderLeftColor: '#4ecca3',
  },
  entryError: {
    borderLeftColor: '#e74c3c',
  },
  entryHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 6,
  },
  entryType: {
    color: '#7f8c8d',
    fontSize: 11,
    fontWeight: '600',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  entryTime: {
    color: '#7f8c8d',
    fontSize: 11,
  },
  entryMessage: {
    color: '#ecf0f1',
    fontSize: 13,
    lineHeight: 18,
    marginBottom: 8,
  },
  errorText: {
    color: '#e74c3c',
  },
  statusBadge: {
    alignSelf: 'flex-start',
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 6,
  },
  badgeSuccess: {
    backgroundColor: 'rgba(78, 204, 163, 0.15)',
  },
  badgeError: {
    backgroundColor: 'rgba(231, 76, 60, 0.15)',
  },
  statusText: {
    fontSize: 10,
    fontWeight: '700',
    color: '#4ecca3',
    textTransform: 'uppercase',
  },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingVertical: 60,
  },
  emptyIcon: {
    fontSize: 36,
    color: '#34495e',
    marginBottom: 12,
  },
  emptyText: {
    color: '#7f8c8d',
    fontSize: 16,
    fontWeight: '600',
  },
  emptySubtext: {
    color: '#4a5568',
    fontSize: 13,
    marginTop: 6,
    textAlign: 'center',
    paddingHorizontal: 40,
  },
});
