import React from 'react';
import { StyleSheet, Switch, Text, View } from 'react-native';

export default function StatusCard({
  title,
  subtitle,
  enabled,
  onToggle,
  statusColor,
}) {
  return (
    <View style={styles.card}>
      <View style={styles.indicator}>
        <View style={[styles.dot, { backgroundColor: statusColor || '#4ecca3' }]} />
      </View>
      <View style={styles.textContainer}>
        <Text style={styles.title}>{title}</Text>
        {subtitle ? <Text style={styles.subtitle}>{subtitle}</Text> : null}
      </View>
      {onToggle !== undefined && (
        <Switch
          value={enabled}
          onValueChange={onToggle}
          trackColor={{ false: '#2c3e50', true: '#4ecca3' }}
          thumbColor={enabled ? '#fff' : '#95a5a6'}
          ios_backgroundColor="#2c3e50"
        />
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#1a1a2e',
    borderRadius: 14,
    padding: 16,
    marginBottom: 12,
  },
  indicator: {
    marginRight: 14,
  },
  dot: {
    width: 10,
    height: 10,
    borderRadius: 5,
  },
  textContainer: {
    flex: 1,
  },
  title: {
    color: '#ecf0f1',
    fontSize: 15,
    fontWeight: '600',
  },
  subtitle: {
    color: '#7f8c8d',
    fontSize: 12,
    marginTop: 2,
  },
});
