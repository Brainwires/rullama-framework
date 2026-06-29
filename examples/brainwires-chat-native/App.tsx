import React from 'react';
import { StatusBar, useColorScheme } from 'react-native';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { GluestackUIProvider } from './src/components/ui/gluestack-ui-provider';
import { Home } from './src/screens/Home';
import './global.css';

function App() {
  const isDarkMode = useColorScheme() === 'dark';
  return (
    <SafeAreaProvider>
      <StatusBar barStyle={isDarkMode ? 'light-content' : 'dark-content'} />
      <GluestackUIProvider mode="system">
        <Home />
      </GluestackUIProvider>
    </SafeAreaProvider>
  );
}

export default App;
