import React from 'react';
import { View, useColorScheme, type ViewProps } from 'react-native';
import { OverlayProvider } from '@gluestack-ui/overlay';
import { ToastProvider } from '@gluestack-ui/toast';
import { config } from './config';

type ModeType = 'light' | 'dark' | 'system';

type Props = { mode?: ModeType; children?: React.ReactNode } & ViewProps;

export function GluestackUIProvider({ mode = 'system', children, style, ...rest }: Props) {
  const colorScheme = useColorScheme();
  const resolved: 'light' | 'dark' =
    mode === 'system' ? (colorScheme === 'dark' ? 'dark' : 'light') : mode;
  return (
    <View
      style={[config[resolved], { flex: 1, width: '100%', height: '100%' }, style]}
      {...rest}
    >
      <OverlayProvider>
        <ToastProvider>{children}</ToastProvider>
      </OverlayProvider>
    </View>
  );
}
