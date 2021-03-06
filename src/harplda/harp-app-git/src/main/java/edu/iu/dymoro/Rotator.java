/*
 * Copyright 2013-2017 Indiana University
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package edu.iu.dymoro;

import java.util.LinkedList;
import java.util.List;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.mapred.CollectiveMapper;

import edu.iu.harp.partition.Partition;
import edu.iu.harp.partition.Table;
import edu.iu.harp.resource.Simple;
import edu.iu.harp.schstatic.StaticScheduler;

public class Rotator<P extends Simple> {

  protected static final Log LOG =
    LogFactory.getLog(Rotator.class);

  private StaticScheduler<Integer, List<Partition<P>>[], RotateTask<P>> rotation;
  private final int numTasks;

  public Rotator(Table<P>[] tableMap,
    int numSplits, boolean randomSplit,
    CollectiveMapper<?, ?, ?, ?> mapper,
    int[] orders, String contextName) {
    numTasks = tableMap.length;
    List<RotateTask<P>> rotateTasks =
      new LinkedList<>();
    for (int i = 0; i < numTasks; i++) {
      rotateTasks.add(new RotateTask<>(
        tableMap[i], numSplits, randomSplit,
        mapper, orders, contextName));
    }
    rotation = new StaticScheduler<>(rotateTasks);
  }

  public List<Partition<P>>[]
    getSplitMap(int taskID) {
    List<Partition<P>>[] splitMap = null;
    if (rotation.hasOutput(taskID)) {
      splitMap = rotation.waitForOutput(taskID);
    } else {
      splitMap =
        rotation.getTask(taskID).getSplitMap();
    }
    return splitMap;
  }

  public void rotate(int taskID) {
    rotation.submit(taskID, 1);
  }

  public void start() {
    rotation.start();
  }

  public void pause() {
    rotation.pause();
  }

  public void stop() {
    rotation.stop();
  }

  public void setRandomSplit(boolean b) {
    for (int i = 0; i < numTasks; i++) {
      rotation.getTask(i).setRandomSplit(b);
    }
  }
}
