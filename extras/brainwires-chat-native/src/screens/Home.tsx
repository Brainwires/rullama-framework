import React, { useEffect, useState } from 'react';
import { Text, View } from 'react-native';
import { frameworkVersion } from '../lib/bridge';

export function Home() {
  const [version, setVersion] = useState<string>('loading…');

  useEffect(() => {
    frameworkVersion()
      .then(setVersion)
      .catch((e) => setVersion(`error: ${e?.message ?? String(e)}`));
  }, []);

  return (
    <View className="flex-1 items-center justify-center bg-white dark:bg-neutral-900 p-6">
      <Text className="text-2xl font-bold text-neutral-900 dark:text-neutral-50 mb-2">
        Brainwires Chat
      </Text>
      <Text className="text-base text-neutral-600 dark:text-neutral-400">
        framework: {version}
      </Text>
    </View>
  );
}
