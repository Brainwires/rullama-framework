/// <reference lib="DOM" />
import { AppRegistry } from 'react-native';
import App from './App';
import appJson from './app.json';
import './global.css';

const appName = (appJson as { name?: string }).name ?? 'BrainwiresChatNative';

AppRegistry.registerComponent(appName, () => App);
AppRegistry.runApplication(appName, {
  rootTag: document.getElementById('root'),
});
